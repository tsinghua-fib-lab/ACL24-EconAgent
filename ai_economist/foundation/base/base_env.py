# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import random
from abc import ABC, abstractmethod
from copy import deepcopy

import numpy as np

from ai_economist.foundation.agents import agent_registry
from ai_economist.foundation.base.registrar import Registry
from ai_economist.foundation.base.world import World
from ai_economist.foundation.components import component_registry
from ai_economist.foundation.entities import (
    endogenous_registry,
    landmark_registry,
    resource_registry,
)


class BaseEnvironment(ABC):
    """
    Base Environment class. Should be used as the parent class for Scenario classes.
    Instantiates world, agent, and component objects.

    Provides Gym-style API for resetting and stepping:
        obs                  <-- env.reset()
        obs, rew, done, info <-- env.step(actions)

    Also provides Gym-style API for controlling random behavior:
        env.seed(seed) # Sets numpy and built-in RNG seeds to seed

    Reference: OpenAI Gym [https://github.com/openai/gym]

    Environments in this framework are instances of Scenario classes (which are built
    as extensions of BaseEnvironment). A Scenario must implement the following
    abstract methods (method docstrings provide detailed explanations):
        reset_starting_layout
        reset_agent_states
        scenario_step
        generate_observations
        compute_reward

    Scenario classes define their own passive dynamics--that is, dynamics that do not
    depend on agent actions--and supply observations. It is up to the Scenario class
    to handle reward.

    Interactions with the environment are handled through components, which define
    actions that agents can perform. Components are defined through distinct
    Component classes (which extend BaseComponent [see base_component.py]) and must
    be included in the components_registry in order to be used (see below).
    Components influence the environment dynamics through effects they have on
    agent/world states. They also (optionally) supply observations.

    The actions available to the agents, observations they receive, the dynamics of
    the environment, and the rewards depend of the choice of which Scenario class and
    Component class(es) to use.

    In multi_action_mode, an agent may choose an action for each of the action
    subspaces defined by the included Component classes. A Component can define 0, 1,
    or several action subspaces for a given agent type. If not using
    multi_action_mode, these action subspaces are combined into a single action space
    and the agent may select one action within this aggregated space.

    For additional detail regarding actions and action subspaces, see the
    BaseComponent class in base_component.py.

    There are 2 types of agents: mobile agents and the planner agent. There can be
    two or more mobile agents and a single planner agent. Conceptually, mobile agents
    represent the individual actors in the economic simulation while the planner
    agent represents a social planner that sets macroeconomic policy.

    This environment framework makes extensive use of Python classes. Scenarios,
    Components, Agents, and environment entities such as Resources, Landmarks,
    and Endogenous variables are all implemented as classes. These classes are
    accessed via registries. See top example.

    Example:
        from ai_economist import foundation
        # foundation.scenarios  <-- Scenario class registry
        # foundation.components <-- Component class registry
        # foundation.agents     <-- Agent class registry
        # foundation.resources  <-- Resource class registry
        # foundation.landmarks  <-- Landmark class registry
        # foundation.endogenous <-- Endogenous class registry

        # see ../scenarios/simple_wood_and_stone/dynamic_layout.py
        UniScenarioClass = foundation.scenarios.get("uniform/simple_wood_and_stone")

        # see ../components/build.py and ../components/move.py
        BuildComponentClass  = foundation.components.get("Build")
        GatherComponentClass = foundation.components.get("Gather")

    Example:
        from ai_economist import foundation
        from ai_economist.foundation.base.base_env import BaseEnvironment

        ScenarioClass = foundation.scenarios.get(...)
        assert issubclass(ScenarioClass, BaseEnvironment)

        env = ScenarioClass(
            components=[
                ("Build", {"payment": 20}),
                ("Gather", {"move_labor": 1.0, "collect_labor": 2.0}),
            ],
            n_agents=20,
            world_size=[25, 25],
        )

        obs = env.reset()

        actions = {agent.idx: ... for agent in env.all_agents}
        obs, rew, done, info = env.step(actions)

    Args:
        components (list): A list of tuples ("Component Name", {Component kwargs}) or
            list of dicts {"Component Name": {Component kwargs}} specifying the
            components that the instantiated environment will include.
            "Component Name" must be a string matching the name of a registered
            Component class.
            {Component kwargs} must be a dictionary of kwargs that can be passed as
            arguments to the Component class with name "Component Name".
            Resetting, stepping, and observation generation will be carried out in
            the order in which components are listed. This should be considered,
            as re-ordering the components list may impact the dynamics of the
            environment.
        n_agents (int): The number of mobile agents (does not include planner).
            Number of agents must be > 1.
        world_size (list): A length-2 list specifying the dimensions of the 2D world.
            Interpreted as [height, width].
        episode_length (int): Number of timesteps in a single episode.
        multi_action_mode_agents (bool): Whether mobile agents use multi_action_mode.
        multi_action_mode_planner (bool): Whether the planner uses multi_action_mode.
        flatten_observations (bool): Whether to preprocess observations by
            concatenating all scalar/vector observation subfields into a single
            "flat" observation field. If not, return observations as minimally
            processed dictionaries.
        flatten_masks (bool): Whether to flatten action masks into a single array or
            to keep as a {"action_subspace_name": action_subspace_mask} dictionary.
            For integration with deep RL, it is helpful to set this to True, for the
            purpose of action masking: flattened masks have the same semantics as
            policy logits.
        allow_observation_scaling (bool): Whether to enable certain observation
            fields to be scaled to a range better suited for deep RL.
        dense_log_frequency (int): [optional] How often (in completed episodes) to
            create a dense log while playing an episode. By default, dense logging is
            turned off (dense_log_frequency=None). If dense_log_frequency=20,
            a dense log will be created when the total episode count is a multiple of
            20.
            Dense logs provide a log of agent states, actions, and rewards at each
            timestep of an episode. They also log world states at a coarser timescale
            (see below). Component classes optionally contribute additional
            information to the dense log.
            Note: dense logging is time consuming (especially with many agents).
        world_dense_log_frequency (int): When dense logging, how often (in timesteps) to
            log a snapshot of the world state. If world_dense_log_frequency=50
            (the default), the world state will be included in the dense log for
            timesteps where t is a multiple of 50.
            Note: More frequent world snapshots increase the dense log memory footprint.
        seed (int, optional): If provided, sets the numpy and built-in random number
            generator seeds to seed. You can control the seed after env construction
            using the 'seed' method.
    """

    # The name associated with this Scenario class (must be unique)
    # Note: This is what will identify the Scenario class in the scenario registry.
    name = ""

    # The (sub)classes of agents that this scenario applies to
    agent_subclasses = []

    # The (non-agent) game entities that are expected to be in play
    required_entities = None  # Replace with list or tuple (can be empty)

    def __init__(
        self,
        components=None,
        n_agents=None,
        world_size=None,
        episode_length=1000,
        multi_action_mode_agents=False,
        multi_action_mode_planner=True,
        flatten_observations=True,
        flatten_masks=True,
        allow_observation_scaling=True,
        dense_log_frequency=None,
        world_dense_log_frequency=50,
        collate_agent_step_and_reset_data=False,
        seed=None,
    ):

        # Make sure a name was declared by child class
        assert self.name

        # Make sure the agent_subclasses was declared by child class
        # and does not create potential conflicts
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

        # Make sure the required_entities was declared by child class
        # (will typecheck later)
        assert isinstance(self.required_entities, (tuple, list))

        # World size must be a tuple or list of length 2,
        # specifying [Height, Width] of the game map
        assert isinstance(world_size, (tuple, list))
        assert len(world_size) == 2
        self.world_size = world_size

        # Number of agents must be an integer and there must be at least 2 agents
        assert isinstance(n_agents, int)
        assert n_agents >= 2
        self.n_agents = n_agents

        # Foundation assumes there's only a single planner
        n_planners = 1
        self.num_agents = (
            n_agents + n_planners
        )  # used in the warp_drive env wrapper (+ 1 for the planner)

        # Components must be a tuple/list where each element is either a...
        #   tuple: ('Component Name', {Component kwargs})
        #   dict : {'Component Name': {Component kwargs}}
        assert isinstance(components, (tuple, list))

        def spec_is_valid(spec):
            """Return True if component specification is validly configured."""
            if isinstance(spec, (tuple, list)):
                if len(spec) != 2:
                    return False
                return isinstance(spec[0], str) and isinstance(spec[1], dict)
            if isinstance(spec, dict):
                if len(spec) != 1:
                    return False
                key_is_str = isinstance(list(spec.keys())[0], str)
                val_is_dict = isinstance(list(spec.values())[0], dict)
                return key_is_str and val_is_dict
            return False

        assert all(spec_is_valid(component) for component in components)

        self._episode_length = int(episode_length)
        assert self._episode_length >= 1

        # Can an agent/planner execute multiple actions (1 per action subspace) per
        # timestep (=True) or just one action (=False)
        self.multi_action_mode_agents = bool(multi_action_mode_agents)
        self.multi_action_mode_planner = bool(multi_action_mode_planner)

        # Whether to allow the world to scale observations
        self._allow_observation_scaling = bool(allow_observation_scaling)

        # Whether to flatten the observation dictionaries before returning them
        # Note: flattened observations are still returned as dictionaries, but with
        # all scalar/vector observation fields concatenated into a single "flat" field.
        self._flatten_observations = bool(flatten_observations)

        # Whether to flatten the mask dictionaries before putting them in the obs
        self._flatten_masks = bool(flatten_masks)

        # How often (in episode completions) to create a dense log
        self._dense_log_this_episode = False
        if dense_log_frequency is None:  # Only create a dense log
            # if manually specified during reset
            self._create_dense_log_every = None
        else:  # Create a dense log every dense_log_frequency episodes
            self._create_dense_log_every = int(dense_log_frequency)
            assert self._create_dense_log_every >= 1

        # How often (in timesteps) to snapshot the world map when creating the denselog
        self._world_dense_log_frequency = int(world_dense_log_frequency)
        assert self._world_dense_log_frequency >= 1

        # Seed control
        if seed is not None:
            self.seed(seed)

        # Initialize the set of entities used in the game that's being created.
        # Coin and Labor are always included.
        self._entities = {
            "resources": ["Coin", 'Products'],
            "landmarks": [],
            "endogenous": ["Labor", "Consumption Rate", "Investment Rate", 'Saving Rate'],
        }
        self._register_entities(self.required_entities)

        # Register all the components to get the entities they rely on.
        self._components = []
        self._components_dict = {}
        self._shorthand_lookup = {}
        component_classes = []
        for component_spec in components:
            if isinstance(component_spec, (tuple, list)):
                component_name, component_config = component_spec
            elif isinstance(component_spec, dict):
                assert len(component_spec) == 1
                component_name = list(component_spec.keys())[0]
                component_config = list(component_spec.values())[0]
            else:
                raise TypeError
            component_cls = component_registry.get(component_name)
            self._register_entities(component_cls.required_entities)
            component_classes.append([component_cls, component_config])

        # Initialize the world object (contains agents and world map),
        # now that we know all the entities we'll use.
        self.world = World(
            self.world_size,
            self.n_agents,
            self.resources,
            self.landmarks,
            self.multi_action_mode_agents,
            self.multi_action_mode_planner,
        )

        # Initialize the component objects.
        for component_cls, component_kwargs in component_classes:
            component_object = component_cls(
                self.world,
                self._episode_length,
                inventory_scale=self.inv_scale,
                **component_kwargs
            )
            self._components.append(component_object)
            self._components_dict[component_object.name] = component_object
            self._shorthand_lookup[component_object.shorthand] = component_object

        # Register the components with the agents
        # to finish setting up their state/action spaces.
        for agent in self.world.agents:
            agent.register_inventory(self.resources)
            agent.register_consumption(self.resources)
            agent.register_investment(self.resources)
            agent.register_saving(self.resources)
            agent.register_income(self.resources)
            agent.register_endogenous(self.endogenous)
            agent.register_components(self._components)
        self.world.planner.register_inventory(self.resources)
        # self.world.planner.register_consumption(self.resources)
        # self.world.planner.register_investment(self.resources)
        # self.world.planner.register_saving(self.resources)
        self.world.planner.register_components(self._components)

        self._agent_lookup = {str(agent.idx): agent for agent in self.all_agents}

        self._completions = 0

        self._last_ep_metrics = None

        # For dense logging
        self._dense_log = {
            "world": [],
            "states": [],
            "actions": [],
            "rewards": [],
        }
        self._last_ep_dense_log = self.dense_log.copy()

        # For episode replay
        self._replay_log = {"reset": dict(seed_state=None), "step": []}
        self._last_ep_replay_log = self.replay_log.copy()

        self._packagers = {}

        # To collate all the agents ('0', '1', ...) data during reset and step
        # into a single agent with index 'a'
        self.collate_agent_step_and_reset_data = collate_agent_step_and_reset_data

    def _register_entities(self, entities):
        for entity in entities:
            if resource_registry.has(entity):
                if entity not in self._entities["resources"]:
                    self._entities["resources"].append(entity)
            elif landmark_registry.has(entity):
                if entity not in self._entities["landmarks"]:
                    self._entities["landmarks"].append(entity)
            elif endogenous_registry.has(entity):
                if entity not in self._entities["endogenous"]:
                    self._entities["endogenous"].append(entity)
            else:
                raise KeyError("Unknown entity: {}".format(entity))

    # Properties
    # ----------

    @property
    def episode_length(self):
        """Length of an episode, in timesteps."""
        return int(self._episode_length)

    @property
    def inv_scale(self):
        """Scale value to be used for inventory scaling. 1 if no scaling enabled."""
        return 0.01 if self._allow_observation_scaling else 1

    @property
    def resources(self):
        """List of resources managed by this environment instance."""
        return sorted(list(self._entities["resources"]))

    @property
    def landmarks(self):
        """List of landmarks managed by this environment instance."""
        return sorted(list(self._entities["landmarks"]))

    @property
    def endogenous(self):
        """List of endogenous quantities managed by this environment instance."""
        return sorted(list(self._entities["endogenous"]))

    @property
    def all_agents(self):
        """List of mobile agents and the planner agent."""
        return self.world.agents + [self.world.planner]

    @property
    def previous_episode_metrics(self):
        """Metrics from the end of the last completed episode."""
        return self._last_ep_metrics

    @property
    def metrics(self):
        """The combined metrics yielded by the scenario and the components."""
        metrics = self.scenario_metrics() or {}

        for component in self._components:
            m_metrics = component.get_metrics()
            if not m_metrics:
                continue
            for k, v in m_metrics.items():
                metrics["{}/{}".format(component.shorthand, k)] = v

        return metrics

    @property
    def components(self):
        """The list of components associated with this scenario."""
        return self._components

    @property
    def dense_log(self):
        """The contents of the current (potentially incomplete) dense log."""
        return self._dense_log

    @property
    def replay_log(self):
        """The contents of the current (potentially incomplete) replay log."""
        return self._replay_log

    @property
    def previous_episode_dense_log(self):
        """Dense log from the last completed episode that was being logged."""
        return self._last_ep_dense_log

    @property
    def previous_episode_replay_log(self):
        """
        Replay log from the last completed episode. Serves as a compact encoding of
        an episode by allowing the episode to be perfectly reproduced.

        Examples:
            # replay log of the episode to be reproduced
            replay_log = env.previous_episode_replay_log

            # recover episode metrics and dense log via replay
            _ = env.reset(force_dense_logging=True, **replay_log['reset'])
            for replay_step in replay_log['step']:
                _ = env.step(**replay_step)
            dense_log = env.previous_episode_dense_log
            metrics = env.previous_episode_metrics
        """
        return self._last_ep_replay_log

    @property
    def generate_rewards(self):
        """Compute the rewards for each agent."""
        return self._generate_rewards

    # Seed control
    # -----------------

    @staticmethod
    def seed(seed):
        """Sets the numpy and built-in random number generator seed.

        Args:
            seed (int, float): Seed value to use. Must be > 0. Converted to int
                internally if provided value is a float.
        """
        assert isinstance(seed, (int, float))
        seed = int(seed)
        assert seed > 0

        np.random.seed(seed)
        random.seed(seed)

    # Getters & Setters
    # -----------------

    def get_component(self, component_name):
        """
        Get the component object instance wrapped in the environment.

        Args:
            component_name (str): Name or shorthand name of the Component class to get.
                Must correspond to a name or shorthand of one of the components that
                is included in this environment instance.

        Returns:
            component (BaseComponent object)
        """
        if component_name not in self._components_dict:
            if component_name not in self._shorthand_lookup:
                raise KeyError(
                    "No component with name or shorthand name {} found; "
                    "registered components are:\n".format(component_name)
                    + "\n\t".join(list(self._components_dict.keys()))
                )
            return self._shorthand_lookup[component_name]
        return self._components_dict[component_name]

    def get_agent(self, agent_idx):
        """
        Get the agent object instance with idx agent_idx.

        Args:
            agent_idx (int or str): Identifier of the agent to return. Must match the
                idx property of one of the agent objects in self.all_agents.

        Returns:
            agent (BaseAgent object)
        """
        agent = self._agent_lookup.get(str(agent_idx), None)
        if agent is None:
            raise ValueError("No agent with associated index {}".format(agent_idx))
        return agent

    def set_agent_component_action(self, agent_idx, component_name, action):
        """
        Set agent with idx <agent_idx> to take action <action> for the action
        subspace with name <component_name>

        Args:
            agent_idx (int or str): Identifier of the agent taking the action. Must
                match the idx property of one of the agent objects in self.all_agents.
            component_name (str): Name of the action subspace to set the action value
                of.
            action (int): Index of the chosen action.
        """
        agent = self.get_agent(agent_idx)
        agent.set_component_action(component_name, action)

    def parse_actions(self, action_dictionary):
        """Put actions into the appropriate agent's action buffer"""
        for agent_idx, agent_actions in action_dictionary.items():
            agent = self.get_agent(agent_idx)
            agent.parse_actions(agent_actions)

    # Core control of environment execution
    # -------------------------------------

    @staticmethod
    def _build_packager(sub_obs, put_in_both=None):
        """
        Decides which keys-vals should be flattened or not.
        put_in_both: include in both (e.g., 'time')
        """
        if put_in_both is None:
            put_in_both = []
        keep_as_is = []
        flatten = []
        wrap_as_list = {}
        for k, v in sub_obs.items():
            if isinstance(v, np.ndarray):
                multi_d_array = len(v.shape) > 1
            else:
                multi_d_array = False

            if k == "action_mask" or multi_d_array:
                keep_as_is.append(k)
            else:
                flatten.append(k)
                if k in put_in_both:
                    keep_as_is.append(k)

            wrap_as_list[k] = np.isscalar(v)

        flatten = sorted(flatten)

        return keep_as_is, flatten, wrap_as_list

    @staticmethod
    def _package(obs_dict, keep_as_is, flatten, wrap_as_list):
        new_obs = {k: obs_dict[k] for k in keep_as_is}
        if len(flatten) == 1:
            k = flatten[0]
            o = obs_dict[k]
            if wrap_as_list[k]:
                o = [o]
            new_obs["flat"] = np.array(o, dtype=np.float32)
        else:
            to_flatten = [
                [obs_dict[k]] if wrap_as_list[k] else obs_dict[k] for k in flatten
            ]
            try:
                new_obs["flat"] = np.concatenate(to_flatten).astype(np.float32)
            except ValueError:
                for k, v in zip(flatten, to_flatten):
                    print(k, np.array(v).shape)
                    print(v)
                    print("")
                raise
        return new_obs

    def _generate_observations(self, flatten_observations=False, flatten_masks=False):
        def recursive_listify(d):
            assert isinstance(d, dict)
            for k, v in d.items():
                if isinstance(v, dict):
                    d[k] = recursive_listify(v)
                elif isinstance(v, (int, float)):
                    d[k] = v
                elif isinstance(v, (list, tuple, set)):
                    d[k] = list(v)
                elif isinstance(v, (np.ndarray, np.integer, np.floating)):
                    d[k] = v.tolist()
                else:
                    raise NotImplementedError(
                        "Not clear how to handle {} with type {}".format(k, type(v))
                    )
                if isinstance(d[k], list) and len(d[k]) == 1:
                    d[k] = d[k][0]
            return d

        # Initialize empty observations
        if self.collate_agent_step_and_reset_data:
            obs = {"a": {}, "p": {}}
        else:
            obs = {str(agent.idx): {} for agent in self.all_agents}
        agent_wise_planner_obs = {
            "p" + str(agent.idx): {} for agent in self.world.agents
        }

        # Get/process observations generated by the scenario
        world_obs = {str(k): v for k, v in self.generate_observations().items()}
        time_scale = self.episode_length if self._allow_observation_scaling else 1.0
        for idx, o in world_obs.items():
            if idx in obs:
                obs[idx].update({"world-" + k: v for k, v in o.items()})
                if self.collate_agent_step_and_reset_data and idx == "a":
                    obs[idx]["time"] = np.array(
                        [
                            self.world.timestep / time_scale
                            for _ in range(self.world.n_agents)
                        ]
                    )
                else:
                    obs[idx]["time"] = [self.world.timestep / time_scale]
            elif idx in agent_wise_planner_obs:
                agent_wise_planner_obs[idx].update(
                    {"world-" + k: v for k, v in o.items()}
                )
            else:
                raise KeyError

        # Get/process observations generated by the components
        for component in self._components:
            for idx, o in component.obs().items():
                if idx in obs:
                    obs[idx].update({component.name + "-" + k: v for k, v in o.items()})
                elif idx in agent_wise_planner_obs:
                    agent_wise_planner_obs[idx].update(
                        {component.name + "-" + k: v for k, v in o.items()}
                    )
                else:
                    raise KeyError
        # print('before flat, agent 0: ', obs['0'], '\n')        
        # print('before flat, planner: ', obs['p'], '\n')
        # Process the observations
        if flatten_observations:
            for o_dict in [obs, agent_wise_planner_obs]:
                for aidx, aobs in o_dict.items():
                    if not aobs:
                        continue
                    if aidx not in self._packagers:
                        self._packagers[aidx] = self._build_packager(
                            aobs, put_in_both=["time"]
                        )
                    try:
                        o_dict[aidx] = self._package(aobs, *self._packagers[aidx])
                    except ValueError:
                        print("Error when packaging obs.")
                        print("Agent index: {}\nRaw obs: {}\n".format(aidx, aobs))
                        raise

        for k, v in agent_wise_planner_obs.items():
            if len(v) > 0:
                obs[self.world.planner.idx][k] = (
                    v["flat"] if flatten_observations else v
                )

        # Get each agent's action masks and incorporate them into the observations
        for aidx, amask in self._generate_masks(flatten_masks=flatten_masks).items():
            obs[aidx]["action_mask"] = amask

        return obs

    def _generate_masks(self, flatten_masks=True):
        if self.collate_agent_step_and_reset_data:
            masks = {"a": {}, "p": {}}
        else:
            masks = {agent.idx: {} for agent in self.all_agents}
        for component in self._components:
            # Use the component's generate_masks method to get action masks
            component_masks = component.generate_masks(completions=self._completions)

            for idx, mask in component_masks.items():
                if isinstance(mask, dict):
                    for sub_action, sub_mask in mask.items():
                        masks[idx][
                            "{}.{}".format(component.name, sub_action)
                        ] = sub_mask
                else:
                    masks[idx][component.name] = mask

        if flatten_masks:
            if self.collate_agent_step_and_reset_data:
                flattened_masks = {}
                for agent_id in masks.keys():
                    if agent_id == "a":
                        multi_action_mode = self.multi_action_mode_agents
                        no_op_mask = np.ones((1, self.n_agents))
                    elif agent_id == "p":
                        multi_action_mode = self.multi_action_mode_planner
                        no_op_mask = [1]
                    mask_dict = masks[agent_id]
                    list_of_masks = []
                    if not multi_action_mode:
                        list_of_masks.append(no_op_mask)
                    for m in mask_dict.keys():
                        if multi_action_mode:
                            list_of_masks.append(no_op_mask)
                        list_of_masks.append(mask_dict[m])
                    flattened_masks[agent_id] = np.concatenate(
                        list_of_masks, axis=0
                    ).astype(np.float32)
                return flattened_masks
            return {
                str(agent.idx): agent.flatten_masks(masks[agent.idx])
                for agent in self.all_agents
            }
        return {
            str(agent_idx): {
                k: np.array(v, dtype=np.uint8).tolist()
                for k, v in masks[agent_idx].items()
            }
            for agent_idx in list(masks.keys())
        }

    def _generate_rewards(self):
        rew = self.compute_reward()
        assert isinstance(rew, dict)
        return {str(k): v for k, v in rew.items()}

    def _finalize_logs(self):
        self._last_ep_replay_log = self._replay_log
        self._last_ep_metrics = self.metrics

        if not self._dense_log_this_episode:
            return

        def recursive_cast(d):
            if isinstance(d, (list, tuple, set)):
                new_d = [recursive_cast(v_) for v_ in d]
                return new_d
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, (list, tuple, set, dict)):
                        d[k] = recursive_cast(v)
                    elif isinstance(v, (int, float, str)):
                        d[k] = v
                    elif isinstance(v, (np.ndarray, np.integer, np.floating)):
                        d[k] = v.tolist()
                    else:
                        raise NotImplementedError(
                            "Not clear how to handle {} with type {}".format(k, type(v))
                        )
                return d
            if isinstance(d, (int, float, str)):
                return d
            if isinstance(d, (np.ndarray, np.integer, np.floating)):
                return d.tolist()
            raise NotImplementedError(
                "Not clear how to handle {} with type {}".format(d, type(d))
            )

        self._dense_log["world"].append(deepcopy(self.world.maps.state_dict))
        self._dense_log["world"][-1].update({'Price': self.world.price[-1], '#Products': self.world.total_products})
        if (self.world.timestep % self.world.period == 0):
            year = self.world.timestep//self.world.period - 1
            self._dense_log["world"][-1].update({'Interest Rate': self.world.interest_rate[-1], 'Unemployment Rate': self.world.unemployment[year]/self.world.period/self.world.n_agents, \
                                                'Real GDP': self.world.real_gdp[year], 'Nominal GDP': self.world.nominal_gdp[year]})
            if (self.world.timestep > self.world.period):
                self._dense_log["world"][-1].update({'Price Inflation': self.world.inflation[-1], 'Unemployment Rate Growth': self.world.unemployment_rate_inflation[-1], \
                                                    'Wage Inflation': self.world.wage_inflation[-1], 'Nominal GDP Growth': self.world.nominal_gdp_inflation[-1], 'Real GDP Growth': self.world.real_gdp_inflation[-1]})
        self._dense_log["states"].append(
            {str(agent.idx): deepcopy(agent.state) for agent in self.all_agents}
        )

        # Back-fill the log with each component's dense log to complete the aggregate
        # dense log
        for component in self._components:
            component_log = component.get_dense_log()
            if component_log is None:
                continue
            if isinstance(component_log, dict):
                for k, v in component_log.items():
                    self._dense_log[component.shorthand + "-" + k] = v
            elif isinstance(component_log, (tuple, list)):
                self._dense_log[component.shorthand] = list(component_log)
            else:
                raise TypeError

        self._last_ep_dense_log = recursive_cast(self._dense_log)

    def collate_agent_obs(self, obs):
        # Collating observations from all agents
        if "a" in obs:  # already collated!
            return obs
        num_agents = len(obs.keys()) - 1
        obs["a"] = {}
        for key in obs["0"].keys():
            obs["a"][key] = np.stack(
                [obs[str(agent_idx)][key] for agent_idx in range(num_agents)], axis=-1
            )
        for agent_idx in range(num_agents):
            del obs[str(agent_idx)]
        return obs

    def collate_agent_rew(self, rew):
        # Collating rewards from all agents
        if "a" in rew:  # already collated!
            return rew
        num_agents = len(rew.keys()) - 1
        rew["a"] = []
        for agent_idx in range(num_agents):
            rew["a"] += [rew[str(agent_idx)]]
            del rew[str(agent_idx)]
        return rew

    def collate_agent_info(self, info):
        # Collating infos from all agents
        if "a" in info:  # already collated!
            return info
        num_agents = len(info.keys()) - 1
        info["a"] = {}
        for agent_idx in range(num_agents):
            info["a"][str(agent_idx)] = info[str(agent_idx)]
            del info[str(agent_idx)]
        return info

    def reset(self, seed_state=None, force_dense_logging=False, seed=None, options=None):
        """
        Reset the state of the environment to initialize a new episode.

        Arguments:
            seed_state (tuple or list): Optional state that the numpy RNG should be set
                to prior to the reset cycle must be length 5, following the format
                expected by np.random.set_state()
            force_dense_logging (bool): Optional whether to force dense logging to take
                place this episode; default behavior is to do dense logging every
                create_dense_log_every episodes

        Returns:
            obs (dict): A dictionary of {"agent_idx": agent_obs} with an entry for
                each agent receiving observations. The "agent_idx" key identifies the
                agent receiving the observations in the associated agent_obs value,
                which itself is a dictionary. The "agent_idx" key matches the
                agent.idx property for the given agent.
        """
        if seed_state is not None:
            assert isinstance(seed_state, (tuple, list))
            assert len(seed_state) == 5
            seed_state = (
                str(seed_state[0]),
                np.array(seed_state[1], dtype=np.uint32),
                int(seed_state[2]),
                int(seed_state[3]),
                float(seed_state[4]),
            )
            np.random.set_state(seed_state)

        if force_dense_logging:
            self._dense_log_this_episode = True
        elif self._create_dense_log_every is None:
            self._dense_log_this_episode = False
        else:
            self._dense_log_this_episode = (
                self._completions % self._create_dense_log_every
            ) == 0

        # For dense logging
        self._dense_log = {
            "world": [],
            "states": [],
            "actions": [],
            "rewards": [],
        }

        # For episode replay
        self._replay_log = {"reset": dict(seed_state=np.random.get_state()), "step": []}

        # Perform the scenario reset,
        # which includes resetting the world and agent states
        self.reset_starting_layout()
        self.reset_agent_states()

        # Perform the component resets for each registered component
        for component in self._components:
            component.reset()

        # Take any customized reset actions
        self.additional_reset_steps()

        # By default, agents take the NO-OP action for each action space.
        # Reset actions to that default.
        for agent in self.all_agents:
            agent.reset_actions()

        # Reset the timestep counter
        self.world.timestep = 0

        # Produce observations
        obs = self._generate_observations(
            flatten_observations=self._flatten_observations,
            flatten_masks=self._flatten_masks,
        )
        # obs['p']['metrics'] = {k: -1 for k in self.metrics}

        if self.collate_agent_step_and_reset_data:
            obs = self.collate_agent_obs(obs)

        return obs

    def step(self, actions=None, seed_state=None):
        """
        Execute the components, perform the scenario step, collect observations and
        return observations, rewards, dones, and infos.

        Arguments:
            actions (dict): dictionary of {agent_idx: action} with an entry for each
                agent (which may include the planner) that is supplying an action.
                The key identifies which agent the action is associated with. It
                should match that agent's agent.idx property.
                The value indicates which action the agent will take. The environment
                supports two formats for specifying an action, with slightly
                different expectations for multi_action_mode.
                If agent.multi_action_mode, action must be a list of integers
                specifying the chosen action for each action subspace.
                Otherwise, action must be a single integer specifying the chosen
                action (where the action space is the concatenation of the subspaces).
            seed_state (tuple or list): Optional state that the numpy RNG should be set
                to prior to the reset cycle must be length 5, following the format
                expected by np.random.set_state().

        Returns:
            obs (dict): A dictionary of {"agent_idx": agent_obs} with an entry for
                each agent receiving observations. The "agent_idx" key identifies the
                agent receiving the observations in the associated agent_obs value,
                which itself is a dictionary. The "agent_idx" key matches the
                agent.idx property for the given agent.
            rew (dict): A dictionary of {"agent_idx": reward} with an entry for each
                agent that also receives an observation. Each reward value is a scalar.
            done (dict): A dictionary with a single key "__all__". The associated
                value is False when self.world.timestep < self.episode_length and True
                otherwise.
            info (dict): Placeholder dictionary with structure {"agent_idx": {}},
                with the same keys as obs and rew.
        """
        if actions is not None:
            assert isinstance(actions, dict)
            self.parse_actions(actions)

        if seed_state is not None:
            assert isinstance(seed_state, (tuple, list))
            assert len(seed_state) == 5
            seed_state = (
                str(seed_state[0]),
                np.array(seed_state[1], dtype=np.uint32),
                int(seed_state[2]),
                int(seed_state[3]),
                float(seed_state[4]),
            )
            np.random.set_state(seed_state)

        self._replay_log["step"].append(
            dict(actions=actions, seed_state=np.random.get_state())
        )

        if self._dense_log_this_episode:
            self._dense_log["world"].append(
                deepcopy(self.world.maps.state_dict)
                if (self.world.timestep % self._world_dense_log_frequency) == 0
                else {}
            )
            self._dense_log["world"][-1].update({'Price': self.world.price[-1], '#Products': self.world.total_products})
            if (self.world.timestep % self.world.period == 0) and (self.world.timestep > 0):
                year = (self.world.timestep-1)//self.world.period
                self._dense_log["world"][-1].update({'Interest Rate': self.world.interest_rate[-1], 'Unemployment Rate': self.world.unemployment[year]/self.world.period/self.world.n_agents, \
                                                    'Real GDP': self.world.real_gdp[year], 'Nominal GDP': self.world.nominal_gdp[year]})
                if (self.world.timestep > self.world.period):
                    self._dense_log["world"][-1].update({'Price Inflation': self.world.inflation[-1], 'Unemployment Rate Growth': self.world.unemployment_rate_inflation[-1], \
                                                         'Wage Inflation': self.world.wage_inflation[-1], 'Nominal GDP Growth': self.world.nominal_gdp_inflation[-1], 'Real GDP Growth': self.world.real_gdp_inflation[-1]})
            self._dense_log["states"].append(
                {str(agent.idx): deepcopy(agent.state) for agent in self.all_agents}
            )
            self._dense_log["actions"].append(
                {
                    str(agent.idx): {k: v for k, v in agent.action.items() if v > 0}
                    for agent in self.all_agents
                }
            )

        self.world.timestep += 1

        for component in self._components:
            # print(component.name)
            component.component_step()

        self.scenario_step()

        obs = self._generate_observations(
            flatten_observations=self._flatten_observations,
            flatten_masks=self._flatten_masks,
        )
        # obs['p']['metrics'] = {k: -1 for k in self.metrics}
        rew = self._generate_rewards()
        done = {"__all__": self.world.timestep >= self._episode_length}
        # truncateds = {"__all__": self.world.timestep >= self._episode_length}
        info = {k: {} for k in obs.keys()}

        if self._dense_log_this_episode:
            self._dense_log["rewards"].append(rew)

        for agent in self.all_agents:
            agent.reset_actions()

        if done[
            "__all__"
        ]:  # Complete the dense log and stash it as well as the metrics
            self._finalize_logs()
            self._completions += 1
            info['p'].update(self._last_ep_metrics)

        if self.collate_agent_step_and_reset_data:
            obs = self.collate_agent_obs(obs)
            rew = self.collate_agent_rew(rew)
            info = self.collate_agent_info(info)

        # return obs, rew, done, truncateds, info
        return obs, rew, done, info

    # The following methods must be implemented for each scenario
    # -----------------------------------------------------------

    @abstractmethod
    def reset_starting_layout(self):
        """
        Part 1/2 of scenario reset. This method handles resetting the state of the
        environment managed by the scenario (i.e. resource & landmark layout).
        """

    @abstractmethod
    def reset_agent_states(self):
        """
        Part 2/2 of scenario reset. This method handles resetting the state of the
        agents themselves (i.e. inventory, locations, etc.).
        """

    @abstractmethod
    def scenario_step(self):
        """
        Update the state of the world according to whatever rules this scenario
        implements.

        This gets called in the 'step' method (of base_env) after going through each
        component step and before generating observations, rewards, etc.

        This is where things like resource regeneration, income redistribution, etc.,
        can be implemented.
        """

    @abstractmethod
    def generate_observations(self):
        """
        Generate observations associated with this scenario.

        A scenario does not need to produce observations and can provide observations
        for only some agent types; however, for a given agent type, it should either
        always or never yield an observation. If it does yield an observation,
        that observation should always have the same structure/sizes!

        Returns:
            obs (dict): A dictionary of {agent.idx: agent_obs_dict}. In words,
                return a dictionary with an entry for each agent (which can including
                the planner) for which this scenario provides an observation. For each
                entry, the key specifies the index of the agent and the value contains
                its associated observation dictionary.
        """

    @abstractmethod
    def compute_reward(self):
        """
        Apply the reward function(s) associated with this scenario to get the rewards
        from this step.

        Returns:
            rew (dict): A dictionary of {agent.idx: agent_obs_dict}. In words,
                return a  dictionary with an entry for each agent in the environment
                (including the planner). For each entry, the key specifies the index of
                the agent and the value contains the scalar reward earned this timestep.
        """

    # Optional methods for customization
    # ----------------------------------

    def additional_reset_steps(self):
        """
        Extra scenario-specific steps that should be performed at the end of the reset
        cycle.

        For each reset cycle...
            First, reset_starting_layout() and reset_agent_states() will be called.

            Second, <component>.reset() will be called for each registered component.

            Lastly, this method will be called to allow for any final customization of
            the reset cycle.
        """

    def scenario_metrics(self):
        """
        Allows the scenario to generate metrics (collected along with component metrics
        in the 'metrics' property).

        To have the scenario add metrics, this function needs to return a dictionary of
        {metric_key: value} where 'value' is a scalar (no nesting or lists!)
        """
        return


scenario_registry = Registry(BaseEnvironment)
"""The registry for Scenario classes.

This creates a registry object for Scenario classes. This registry requires that all
added classes are subclasses of BaseEnvironment. To make a Scenario class available
through the registry, decorate the class definition with @scenario_registry.add.

Example:
    from ai_economist.foundation.base.base_env
    import BaseEnvironment, scenario_registry

    @scenario_registry.add
    class ExampleScenario(BaseEnvironment):
        name = "Example"
        pass

    assert scenario_registry.has("Example")

    ScenarioClass = scenario_registry.get("Example")
    scenario = ScenarioClass(...)
    assert isinstance(scenario, ExampleScenario)

Notes:
    The foundation package exposes the scenario registry as: foundation.scenarios

    A Scenario class that is defined and registered following the above example will
    only be visible in foundation.scenarios if defined/registered in a file that is
    imported in ../scenarios/__init__.py.
"""
