# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import random

import numpy as np

from collections import deque

from ai_economist.foundation.base.registrar import Registry


class BaseAgent:
    """Base class for Agent classes.

    Instances of Agent classes are created for each agent in the environment. Agent
    instances are stateful, capturing location, inventory, endogenous variables,
    and any additional state fields created by environment components during
    construction (see BaseComponent.get_additional_state_fields in base_component.py).

    They also provide a simple API for getting/setting actions for each of their
    registered action subspaces (which depend on the components used to build
    the environment).

    Args:
        idx (int or str): Index that uniquely identifies the agent object amongst the
            other agent objects registered in its environment.
        multi_action_mode (bool): Whether to allow the agent to take one action for
            each of its registered action subspaces each timestep (if True),
            or to limit the agent to take only one action each timestep (if False).
    """

    name = ""

    def __init__(self, idx=None, multi_action_mode=None):
        assert self.name

        if idx is None:
            idx = 0

        if multi_action_mode is None:
            multi_action_mode = False

        if isinstance(idx, str):
            self._idx = idx
        else:
            self._idx = int(idx)

        self.multi_action_mode = bool(multi_action_mode)
        self.single_action_map = (
            {}
        )  # Used to convert single-action-mode actions to the general format

        self.action = dict()
        self.action_dim = dict()
        self._action_names = []
        self._multi_action_dict = {}
        self._unique_actions = 0
        self._total_actions = 0

        self.state = dict(loc=[0, 0], inventory={}, escrow={}, consumption={}, investment={}, saving={}, income={}, endogenous={})

        self._registered_inventory = False
        self._registered_consumption = False
        self._registered_investment = False
        self._registered_saving = False
        self._registered_income = False
        self._registered_endogenous = False
        self._registered_components = False
        self._noop_action_dict = dict()

        # Special flag to allow logic for multi-action-mode agents
        # that are not given any actions.
        self._passive_multi_action_agent = False

        # If this gets set to true, we can make masks faster
        self._one_component_single_action = False
        self._premask = None

    @property
    def idx(self):
        """Index used to identify this agent. Must be unique within the environment."""
        return self._idx

    def register_inventory(self, resources):
        """Used during environment construction to populate inventory/escrow fields."""
        assert not self._registered_inventory
        for entity_name in resources:
            self.inventory[entity_name] = 0
            self.escrow[entity_name] = 0
        self._registered_inventory = True
        
    def register_consumption(self, resources):
        assert not self._registered_consumption
        for entity_name in resources:
            self.consumption[entity_name] = 0
        self._registered_consumption = True
        
    def register_investment(self, resources):
        assert not self._registered_investment
        for entity_name in resources:
            self.investment[entity_name] = 0
        self._registered_investment = True
    
    def register_saving(self, resources):
        assert not self._registered_saving
        for entity_name in resources:
            self.saving[entity_name] = 0
        self._registered_saving = True
        
    def register_income(self, resources):
        assert not self._registered_income
        for entity_name in resources:
            self.income[entity_name] = 0
        self._registered_income = True

    def register_endogenous(self, endogenous):
        """Used during environment construction to populate endogenous state fields."""
        assert not self._registered_endogenous
        for entity_name in endogenous:
            self.endogenous[entity_name] = 0
        self._registered_endogenous = True

    def _incorporate_component(self, action_name, n):
        extra_n = (
            1 if self.multi_action_mode else 0
        )  # Each sub-action has a NO-OP in multi action mode)
        self.action[action_name] = 0
        self.action_dim[action_name] = n + extra_n
        self._action_names.append(action_name)
        self._multi_action_dict[action_name] = False
        self._unique_actions += 1
        if self.multi_action_mode:
            self._total_actions += n + extra_n
        else:
            for action_n in range(1, n + 1):
                self._total_actions += 1
                self.single_action_map[int(self._total_actions)] = [
                    action_name,
                    action_n,
                ]

    def register_components(self, components):
        """Used during environment construction to set up state/action spaces."""
        assert not self._registered_components
        for component in components:
            n = component.get_n_actions(self.name)
            if n is None:
                continue

            # Most components will have a single action-per-agent, so n is an int
            if isinstance(n, int):
                if n == 0:
                    continue
                self._incorporate_component(component.name, n)

            # They can also internally handle multiple actions-per-agent,
            # so n is an tuple or list
            elif isinstance(n, (tuple, list)):
                for action_sub_name, n_ in n:
                    if n_ == 0:
                        continue
                    if "." in action_sub_name:
                        raise NameError(
                            "Sub-action {} of component {} "
                            "is illegally named.".format(
                                action_sub_name, component.name
                            )
                        )
                    self._incorporate_component(
                        "{}.{}".format(component.name, action_sub_name), n_
                    )

            # If that's not what we got something is funky.
            else:
                raise TypeError(
                    "Received unexpected type ({}) from {}.get_n_actions('{}')".format(
                        type(n), component.name, self.name
                    )
                )

            for k, v in component.get_additional_state_fields(self.name).items():
                self.state[k] = v

        # Currently no actions are available to this agent. Give it a placeholder.
        if len(self.action) == 0 and self.multi_action_mode:
            self._incorporate_component("PassiveAgentPlaceholder", 0)
            self._passive_multi_action_agent = True

        elif len(self.action) == 1 and not self.multi_action_mode:
            self._one_component_single_action = True
            self._premask = np.ones(1 + self._total_actions, dtype=np.float32)

        self._registered_components = True

        self._noop_action_dict = {k: v * 0 for k, v in self.action.items()}

        verbose = False
        if verbose:
            print(self.name, self.idx, "constructed action map:")
            for k, v in self.single_action_map.items():
                print("single action map:", k, v)
            for k, v in self.action.items():
                print("action:", k, v)
            for k, v in self.action_dim.items():
                print("action_dim:", k, v)

    @property
    def action_spaces(self):
        """
        if self.multi_action_mode == True:
            Returns an integer array with length equal to the number of action
            subspaces that the agent registered. The i'th element of the array
            indicates the number of actions associated with the i'th action subspace.
            In multi_action_mode, each subspace includes a NO-OP.
            Note: self._action_names describes which action subspace each element of
            the array refers to.

            Example:
                >> self.multi_action_mode
                True
                >> self.action_spaces
                [2, 5]
                >> self._action_names
                ["Build", "Gather"]
                # [1 Build action + Build NO-OP, 4 Gather actions + Gather NO-OP]

        if self.multi_action_mode == False:
            Returns a single integer equal to the total number of actions that the
            agent can take.

            Example:
                >> self.multi_action_mode
                False
                >> self.action_spaces
                6
                >> self._action_names
                ["Build", "Gather"]
                # 1 NO-OP + 1 Build action + 4 Gather actions.
        """
        if self.multi_action_mode:
            action_dims = []
            for m in self._action_names:
                action_dims.append(np.array(self.action_dim[m]).reshape(-1))
            return np.concatenate(action_dims).astype(np.int32)
        n_actions = 1  # (NO-OP)
        for m in self._action_names:
            n_actions += self.action_dim[m]
        return n_actions

    @property
    def loc(self):
        """2D list of [row, col] representing agent's location in the environment."""
        return self.state["loc"]

    @property
    def endogenous(self):
        """Dictionary representing endogenous quantities (i.e. "Labor").

        Example:
            >> self.endogenous
            {"Labor": 30.25}
        """
        return self.state["endogenous"]

    @property
    def inventory(self):
        """Dictionary representing quantities of resources in agent's inventory.

        Example:
            >> self.inventory
            {"Wood": 3, "Stone": 20, "Coin": 1002.83}
        """
        return self.state["inventory"]

    @property
    def escrow(self):
        """Dictionary representing quantities of resources in agent's escrow.

        https://en.wikipedia.org/wiki/Escrow
        Escrow is used to manage any portion of the agent's inventory that is
        reserved for a particular purpose. Typically, something enters escrow as part
        of a contractual arrangement to disburse that something when another
        condition is met. An example is found in the ContinuousDoubleAuction
        Component class (see ../components/continuous_double_auction.py). When an
        agent creates an order to sell a unit of Wood, for example, the component
        moves one unit of Wood from the agent's inventory to its escrow. If another
        agent buys the Wood, it is moved from escrow to the other agent's inventory. By
        placing the Wood in escrow, it prevents the first agent from using it for
        something else (i.e. building a house).

        Notes:
            The inventory and escrow share the same keys. An agent's endowment refers
            to the total quantity it has in its inventory and escrow.

            Escrow is provided to simplify inventory management but its intended
            semantics are not enforced directly. It is up to Component classes to
            enforce these semantics.

        Example:
            >> self.inventory
            {"Wood": 0, "Stone": 1, "Coin": 3}
        """
        return self.state["escrow"]

    def inventory_to_escrow(self, resource, amount):
        """Move some amount of a resource from agent inventory to agent escrow.

        Amount transferred is capped to the amount of resource in agent inventory.

        Args:
            resource (str): The name of the resource to move (i.e. "Wood", "Coin").
            amount (float): The amount to be moved from inventory to escrow. Must be
                positive.

        Returns:
            Amount of resource actually transferred. Will be less than amount argument
                if amount argument exceeded the amount of resource in the inventory.
                Calculated as:
                    transferred = np.minimum(self.state["inventory"][resource], amount)
        """
        assert amount >= 0
        transferred = float(np.minimum(self.state["inventory"][resource], amount))
        self.state["inventory"][resource] -= transferred
        self.state["escrow"][resource] += transferred
        return float(transferred)

    def escrow_to_inventory(self, resource, amount):
        """Move some amount of a resource from agent escrow to agent inventory.

        Amount transferred is capped to the amount of resource in agent escrow.

        Args:
            resource (str): The name of the resource to move (i.e. "Wood", "Coin").
            amount (float): The amount to be moved from escrow to inventory. Must be
                positive.

        Returns:
            Amount of resource actually transferred. Will be less than amount argument
                if amount argument exceeded the amount of resource in escrow.
                Calculated as:
                    transferred = np.minimum(self.state["escrow"][resource], amount)
        """
        assert amount >= 0
        transferred = float(np.minimum(self.state["escrow"][resource], amount))
        self.state["escrow"][resource] -= transferred
        self.state["inventory"][resource] += transferred
        return float(transferred)
    
    @property
    def consumption(self):
        return self.state["consumption"]
    
    @property
    def investment(self):
        return self.state["investment"]
    
    @property
    def saving(self):
        return self.state["saving"]
    
    @property
    def income(self):
        return self.state["income"]

    def total_endowment(self, resource):
        """Get the combined inventory+escrow endowment of resource.

        Args:
            resource (str): Name of the resource

        Returns:
            The amount of resource in the agents inventory and escrow.

        """
        return self.inventory[resource] + self.escrow[resource]

    def reset_actions(self, component=None):
        """Reset all actions to the NO-OP action (the 0'th action index).

        If component is specified, only reset action(s) for that component.
        """
        if not component:
            self.action.update(self._noop_action_dict)
        else:
            for k, v in self.action.items():
                if "." in component:
                    if k.lower() == component.lower():
                        self.action[k] = v * 0
                else:
                    base_component = k.split(".")[0]
                    if base_component.lower() == component.lower():
                        self.action[k] = v * 0

    def has_component(self, component_name):
        """Returns True if the agent has component_name as a registered subaction."""
        return bool(component_name in self.action)

    def get_random_action(self):
        """
        Select a component at random and randomly choose one of its actions (other
        than NO-OP).
        """
        random_component = random.choice(self._action_names)
        component_action = random.choice(
            list(range(1, self.action_dim[random_component]))
        )
        return {random_component: component_action}

    def get_component_action(self, component_name, sub_action_name=None):
        """
        Return the action(s) taken for component_name component, or None if the
        agent does not use that component.
        """
        if sub_action_name is not None:
            return self.action.get(component_name + "." + sub_action_name, None)
        matching_names = [
            m for m in self._action_names if m.split(".")[0] == component_name
        ]
        if len(matching_names) == 0:
            return None
        if len(matching_names) == 1:
            return self.action.get(matching_names[0], None)
        return [self.action.get(m, None) for m in matching_names]

    def set_component_action(self, component_name, action):
        """Set the action(s) taken for component_name component."""
        if component_name not in self.action:
            raise KeyError(
                "Agent {} of type {} does not have {} registered as a subaction".format(
                    self.idx, self.name, component_name
                )
            )
        if self._multi_action_dict[component_name]:
            self.action[component_name] = np.array(action, dtype=np.int32)
        else:
            self.action[component_name] = int(action)

    def populate_random_actions(self):
        """Fill the action buffer with random actions. This is for testing."""
        for component, d in self.action_dim.items():
            if isinstance(d, int):
                self.set_component_action(component, np.random.randint(0, d))
            else:
                d_array = np.array(d)
                self.set_component_action(
                    component, np.floor(np.random.rand(*d_array.shape) * d_array)
                )

    def parse_actions(self, actions):
        """Parse the actions array to fill each component's action buffers."""
        if self.multi_action_mode:
            assert len(actions) == self._unique_actions, (self.idx, len(actions), self._unique_actions)
            if len(actions) == 1:
                self.set_component_action(self._action_names[0], actions[0])
            else:
                for action_name, action in zip(self._action_names, actions):
                    self.set_component_action(action_name, int(action))

        # Single action mode
        else:
            # Action was supplied as an index of a specific subaction.
            # No need to do any lookup.
            if isinstance(actions, dict):
                if len(actions) == 0:
                    return
                assert len(actions) == 1
                action_name = list(actions.keys())[0]
                action = list(actions.values())[0]
                if action == 0:
                    return
                self.set_component_action(action_name, action)

            # Action was supplied as an index into the full set of combined actions
            else:
                action = int(actions)
                # Universal NO-OP
                if action == 0:
                    return
                action_name, action = self.single_action_map.get(action)
                self.set_component_action(action_name, action)

    def flatten_masks(self, mask_dict):
        """Convert a dictionary of component action masks into a single mask vector."""
        if self._one_component_single_action:
            self._premask[1:] = mask_dict[self._action_names[0]]
            return self._premask

        no_op_mask = [1]

        if self._passive_multi_action_agent:
            return np.array(no_op_mask).astype(np.float32)

        list_of_masks = []
        if not self.multi_action_mode:
            list_of_masks.append(no_op_mask)
        for m in self._action_names:
            if m not in mask_dict:
                raise KeyError("No mask provided for {} (agent {})".format(m, self.idx))
            if self.multi_action_mode:
                list_of_masks.append(no_op_mask)
            list_of_masks.append(mask_dict[m])
        return np.concatenate(list_of_masks).astype(np.float32)


agent_registry = Registry(BaseAgent)
"""The registry for Agent classes.

This creates a registry object for Agent classes. This registry requires that all
added classes are subclasses of BaseAgent. To make an Agent class available through
the registry, decorate the class definition with @agent_registry.add.

Example:
    from ai_economist.foundation.base.base_agent import BaseAgent, agent_registry

    @agent_registry.add
    class ExampleAgent(BaseAgent):
        name = "Example"
        pass

    assert agent_registry.has("Example")

    AgentClass = agent_registry.get("Example")
    agent = AgentClass(...)
    assert isinstance(agent, ExampleAgent)

Notes:
    The foundation package exposes the agent registry as: foundation.agents

    An Agent class that is defined and registered following the above example will
    only be visible in foundation.agents if defined/registered in a file that is
    imported in ../agents/__init__.py.
"""
