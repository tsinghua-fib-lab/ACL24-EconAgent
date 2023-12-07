# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from abc import ABC, abstractmethod

import numpy as np

from ai_economist.foundation.agents import agent_registry
from ai_economist.foundation.base.registrar import Registry
from ai_economist.foundation.base.world import World


class BaseComponent(ABC):
    """
    Base Component class. Should be used as the parent class for Component classes.
    Component instances are used to add some particular dynamics to an environment.
    They also add action spaces through which agents can interact with the
    environment via the component instance.

    Environments expand the agents' state/action spaces by querying:
        get_n_actions
        get_additional_state_fields

    Environments expand their dynamics by querying:
        component_step
        generate_observations
        generate_masks

    Environments expand logging behavior by querying:
        get_metrics
        get_dense_log

    Because they are built as Python objects, component instances can also be
    stateful. Stateful attributes are reset via calls to:
        additional_reset_steps

    The semantics of each method, and how they can be used to construct an instance
    of the Component class, are detailed below.

    Refer to ../components/move.py for an example of a Component class that enables
    mobile agents to move and collect resources in the environment world.
    """

    # The name associated with this Component class (must be unique).
    # Note: This is what will identify the Component class in the component registry.
    name = ""

    # An optional shorthand description of the what the component implements (i.e.
    # "Trading", "Building", etc.). See BaseEnvironment.get_component and
    # BaseEnvironment._finalize_logs to see where this may add convenience.
    # Does not need to be unique.
    component_type = None

    # The (sub)classes of agents that this component applies to
    agent_subclasses = None  # Replace with list or tuple (can be empty)

    # The (non-agent) game entities that are expected to be in play
    required_entities = None  # Replace with list or tuple (can be empty)

    def __init__(self, world, episode_length, inventory_scale=1):
        assert self.name

        assert isinstance(self.agent_subclasses, (tuple, list))
        assert len(self.agent_subclasses) > 0
        if len(self.agent_subclasses) > 1:
            for i in range(len(self.agent_subclasses)):
                for j in range(len(self.agent_subclasses)):
                    if i == j:
                        continue
                    a_i = agent_registry.get(self.agent_subclasses[i])
                    a_j = agent_registry.get(self.agent_subclasses[j])
                    assert not issubclass(a_i, a_j)

        assert isinstance(self.required_entities, (tuple, list))

        self.check_world(world)
        self._world = world

        assert isinstance(episode_length, int) and episode_length > 0
        self._episode_length = episode_length

        self.n_agents = world.n_agents
        self.resources = world.resources
        self.landmarks = world.landmarks

        self.timescale = 1
        assert self.timescale >= 1

        self._inventory_scale = float(inventory_scale)

    @property
    def world(self):
        """The world object of the environment this component instance is part of.

        The world object exposes the spatial/agent states through:
            world.maps     # Reference to maps object representing spatial state
            world.agents   # List of self.n_agents mobile agent objects
            world.planner  # Reference to planner agent object

        See world.py and base_agent.py for additional API details.
        """
        return self._world

    @property
    def episode_length(self):
        """Episode length of the environment this component instance is a part of."""
        return int(self._episode_length)

    @property
    def inv_scale(self):
        """
        Value by which to scale quantities when generating observations.

        Note: This property is set by the environment during construction and
        allows each component instance within the environment to refer to the same
        scaling value. How the value is actually used depends on the implementation
        of get_observations().
        """
        return self._inventory_scale

    @property
    def shorthand(self):
        """The shorthand name, or name if no component_type is defined."""
        return self.name if self.component_type is None else self.component_type

    @staticmethod
    def check_world(world):
        """Validate the world object."""
        assert isinstance(world, World)

    def reset(self):
        """Reset any portion of the state managed by this component."""
        world = self.world
        all_agents = world.agents + [world.planner]
        for agent in all_agents:
            agent.state.update(self.get_additional_state_fields(agent.name))

        # This method allows components to define additional reset steps
        self.additional_reset_steps()

    def obs(self):
        """
        Observation produced by this component, given current world/agents/component
        state.
        """
        # This is mostly just to ensure formatting.
        obs = self.generate_observations()
        assert isinstance(obs, dict)
        obs = {str(k): v for k, v in obs.items()}
        return obs

    # Required methods for implementing components
    # --------------------------------------------

    @abstractmethod
    def get_n_actions(self, agent_cls_name):
        """
        Return the number of actions (not including NO-OPs) for agents of type
        agent_cls_name.

        Args:
            agent_cls_name (str): name of the Agent class for which number of actions
                is being queried. For example, "BasicMobileAgent".

        Returns:
            action_space (None, int, or list): If the component does not add any
                actions for agents of type agent_cls_name, return None. If it adds a
                single action space, return an integer specifying the number of
                actions in the action space. If it adds multiple action spaces,
                return a list of tuples ("action_set_name", num_actions_in_set).
                See below for further detail.

        If agent_class_name type agents do not participate in the component, simply
        return None

        In the next simplest case, the component adds one set of n different actions
        for agents of type agent_cls_name. In this case, return n (as an int). For
        example, if Component implements moving up, down, left, or right for
        "BasicMobileAgent" agents, then Component.get_n_actions('Mobile') should
        return 4.

        If the component adds multiple sets of actions for a given agent type, this
        method should return a list of tuples:
            [("action_set_name_1", n_1), ..., ("action_set_name_M", n_M)],
            where M is the number of different sets of actions, and n_k is the number of
            actions in action set k.
        For example, if Component allows agent 'Planner' to set some tax for each of
        individual Mobile agents, and there are 3 such agents, then:
            Component.get_n_actions('Planner') should return, i.e.,
            [('Tax_0', 10), ('Tax_1', 10), ('Tax_2', 10)],
            where, in this example, the Planner agent can choose 10 different tax
            levels for each Mobile agent.
        """

    @abstractmethod
    def get_additional_state_fields(self, agent_cls_name):
        """
        Return a dictionary of {state_field: reset_val} managed by this Component
        class for agents of type agent_cls_name. This also partially controls reset
        behavior.

        Args:
            agent_cls_name (str): name of the Agent class for which additional states
                are being queried. For example, "BasicMobileAgent".

        Returns:
            extra_state_dict (dict): A dictionary of {"state_field": reset_val} for
                each extra state field that this component adds/manages to agents of
                type agent_cls_name. This extra_state_dict is incorporated into
                agent.state for each agent of this type. Note that the keyed fields
                will be reset to reset_val when the environment is reset.

        If the component has its own internal state, the protocol for resetting that
        should be written into the custom method 'additional_reset_steps()' [see below].

        States that are meant to be internal to the component do not need to be
        registered as agent state fields. Rather, adding to the agent state fields is
        most useful when two or more components refer to or affect the same state. In
        general, however, if the component expects a particular state field to exist,
        it should use return that field (and its reset value) here.
        """

    @abstractmethod
    def component_step(self):
        """
        For all relevant agents, execute the actions specific to this Component class.
        This is essentially where the component logic is implemented and what allows
        components to create environment dynamics.

        If the component expects certain resources/landmarks/entities to be in play,
        it must declare them in 'required_entities' so that they can be registered as
        part of the world and, where appropriate, part of the agent inventory.

        If the component expects non-standard fields to exist in agent.state for one
        or more agent types, that must be reflected in get_additional_state_fields().
        """

    @abstractmethod
    def generate_observations(self):
        """
        Generate observations associated with this Component class.

        A component does not need to produce observations and can provide observations
        for only some agent types; however, for a given environment, the structure of
        the observations returned by this component should be identical between
        subsequent calls to generate_observations. That is, the agents that receive
        observations should remain consistent as should the structure of their
        individual observations.

        Returns:
            obs (dict): A dictionary of {agent.idx: agent_obs_dict}. In words,
                return a dictionary with an entry for each agent (which can include
                the planner) for which this component provides an observation. For each
                entry, the key specifies the index of the agent and the value contains
                its associated observation dictionary.
        """

    @abstractmethod
    def generate_masks(self, completions=0):
        """
        Create action masks to indicate which actions are and are not valid. Actions
        that are valid should be given a value of 1 and 0 otherwise. Do not generate
        a mask for the NO-OP action, which is always available.

        Args:
            completions (int): The number of completed episodes. This is intended to
                be used in the case that actions may be masked or unmasked as part of a
                learning curriculum.

        Returns:
            masks (dict): A dictionary of {agent.idx: mask} with an entry for each
                agent that can interact with this component. See below.


        The expected output parallels the action subspaces defined by get_n_actions():
        The output should be a dictionary of {agent.idx: mask} keyed for all agents
        that take actions via this component.

        For example, say the component defines a set of 4 actions for agents of type
        "BasicMobileAgent" (self.get_n_actions("BasicMobileAgent) --> 4). Because all
        action spaces include a NO-OP action, there are 5 available actions,
        interpreted in this example as: NO-OP (index=0), moving up (index=1),
        down (index=2), left (index=3), or right (index=4). Say also that agent-0 (the
        agent with agent.idx=0) is prevented from moving left but can otherwise move.
        In this case, generate_masks(world)['0'] should point to a length-4 binary
        array, specifically [1, 1, 0, 1]. Note that the mask is length 4 while
        technically 5 actions are available. This is because NO-OP should be ignored
        when constructing masks.

        In the more complex case where the component defines several action sets for
        an agent, say the planner agent (the agent with agent.idx='p'), then
        generate_masks(world)['p'] should point to a dictionary of
        {"action_set_name_m": mask_m} for each of the M action sets associated with
        agent p's type. Each such value, mask_m, should be a binary array whose
        length matches the number of actions in "action_set_name_m".

        The default behavior (below) keeps all actions available. The code gives an
        example of expected formatting.
        """
        world = self.world
        masks = {}
        # For all the agents in the environment
        for agent in world.agents + [world.planner]:
            # Get any action space(s) defined by this component for this agent
            n_actions = self.get_n_actions(agent.name)

            # If no action spaces are defined, just move on.
            if n_actions is None:
                continue

            # If a single action space is defined, n_actions corresponds to the
            # number of (non NO-OP) actions. Return an array of ones of that length,
            # enabling all actions.
            if isinstance(n_actions, (int, float)):
                masks[agent.idx] = np.ones(int(n_actions))

            # If multiple action spaces are defined, n_actions corresponds to the
            # tuple or list giving ("name", N) for each action space, where "name"
            # is the unique name and N is the number of (non NO-OP) actions
            # associated with that action space.
            # Return a dictionary of {"name": length-N ones array}, enabling all
            # actions in all the action spaces.
            elif isinstance(n_actions, (tuple, list)):
                masks[agent.idx] = {
                    sub_name: np.ones(int(sub_n)) for sub_name, sub_n in n_actions
                }

            else:
                raise TypeError

        return masks

    # For non-required customization
    # ------------------------------

    def additional_reset_steps(self):
        """
        Use this method to implement additional steps that the component should
        perform at reset. Useful for resetting internal trackers.

        This method should not return anything.
        """
        return

    def get_metrics(self):
        """
        Returns a dictionary of custom metrics describing the episode through the
        lens of the component.

        For example, if Build is a subclass of BaseComponent that implements building,
        Build.get_metrics() might return a dictionary with terms relating to the
        number of things each agent built.

        Returns:
            metrics (dict or None): A dictionary of {"metric_key": metric_value}
                entries describing the metrics that this component calculates. The
                environment combines scenario metrics with each of the metric
                dictionaries produced by its component instances. metric_value is
                expected to be a scalar.
                By returning None instead of a dictionary, the component is ignored
                by the environment when constructing the full metric report.
        """
        return None

    def get_dense_log(self):
        """
        Return the dense log, either a tuple, list, or dict, of the episode through the
        lens of this component.

        If this component does not yield a dense log, return None (default behavior).
        """
        return None


component_registry = Registry(BaseComponent)
"""The registry for Component classes.

This creates a registry object for Component classes. This registry requires that all
added classes are subclasses of BaseComponent. To make a Component class available
through the registry, decorate the class definition with @component_registry.add.

Example:
    from ai_economist.foundation.base.base_component
    import BaseComponent, component_registry

    @component_registry.add
    class ExampleComponent(BaseComponent):
        name = "Example"
        pass

    assert component_registry.has("Example")

    ComponentClass = component_registry.get("Example")
    component = ComponentClass(...)
    assert isinstance(component, ExampleComponent)

Notes:
    The foundation package exposes the component registry as: foundation.components

    A Component class that is defined and registered following the above example will
    only be visible in foundation.components if defined/registered in a file that is
    imported in ../components/__init__.py.
"""
