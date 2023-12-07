# Copyright (c) 2021, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from ai_economist.foundation.base.base_component import (
    BaseComponent,
    component_registry,
)


@component_registry.add
class SimpleLabor(BaseComponent):
    """
    Allows Agents to select a level of labor, which earns income based on skill.

    Labor is "simple" because this simplfies labor to a choice along a 1D axis. More
    concretely, this component adds 100 labor actions, each representing a choice of
    how many hours to work, e.g. action 50 represents doing 50 hours of work; each
    Agent earns income proportional to the product of its labor amount (representing
    hours worked) and its skill (representing wage), with higher skill and higher labor
    yielding higher income.

    This component is intended to be used with the 'PeriodicBracketTax' component and
    the 'one-step-economy' scenario.

    Args:
        mask_first_step (bool): Defaults to True. If True, masks all non-0 labor
            actions on the first step of the environment. When combined with the
            intended component/scenario, the first env step is used to set taxes
            (via the 'redistribution' component) and the second step is used to
            select labor (via this component).
        payment_max_skill_multiplier (float): When determining the skill level of
            each Agent, sampled skills are clipped to this maximum value.
    """

    name = "SimpleLabor"
    required_entities = ["Coin"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        mask_first_step=True,
        labor_step=1,
        num_labor_hours=168,
        payment_max_skill_multiplier=3,
        pareto_param=4.0,
        scale_obs=True,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # This defines the size of the action space (the max # hours an agent can work).
        self.num_labor_hours = num_labor_hours  # max 100 hours
        self.labor_step = labor_step
        self.n_actions = int(self.num_labor_hours//self.labor_step)
        assert isinstance(mask_first_step, bool)
        self.mask_first_step = mask_first_step
        self.scale_obs = scale_obs

        self.is_first_step = True
        self.common_mask_on = {
            agent.idx: np.ones((self.n_actions,)) for agent in self.world.agents
        }
        self.common_mask_off = {
            agent.idx: np.zeros((self.n_actions,)) for agent in self.world.agents
        }

        # Skill distribution
        self.pareto_param = float(pareto_param)
        assert self.pareto_param > 0
        self.payment_max_skill_multiplier = float(payment_max_skill_multiplier)
        pmsm = self.payment_max_skill_multiplier
        num_agents = len(self.world.agents)
        # Generate a batch (1000) of num_agents (sorted/clipped) Pareto samples.
        pareto_samples = np.random.pareto(self.pareto_param, size=(1000, num_agents))
        clipped_skills = np.minimum(pmsm, (pmsm - 1) * pareto_samples + 1)
        sorted_clipped_skills = np.sort(clipped_skills, axis=1)
        # The skill level of the i-th skill-ranked agent is the average of the
        # i-th ranked samples throughout the batch.
        self.skills = sorted_clipped_skills.mean(axis=0)

    def get_additional_state_fields(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return {"skill": 0, 'expected skill': 0, "production": 0}
        return {}

    def additional_reset_steps(self):
        self.is_first_step = True
        for agent in self.world.agents:
            agent.state["skill"] = self.skills[agent.idx]
            agent.state["expected skill"] = self.skills[agent.idx]
        # self.set_offer()

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return self.n_actions
        return None

    def generate_masks(self, completions=0):
        if self.is_first_step:
            self.is_first_step = False
            if self.mask_first_step:
                return self.common_mask_off

        return self.common_mask_on

    def component_step(self):
        
        agent_labors = []
        for agent in self.world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            if action == 0:  # NO-OP.
                # Agent is not interacting with this component.
                agent_labors.append(0)
                continue

            if 1 <= action <= int(self.num_labor_hours//self.labor_step):  # set reopening phase

                hours_worked = action*self.labor_step  # NO-OP is 0 hours.
                agent.state["endogenous"]["Labor"] = hours_worked
                agent_labors.append(hours_worked)

                payoff = hours_worked * agent.state["skill"]
                agent.income["Coin"] = payoff
                agent.state["production"] += payoff
                agent.inventory["Coin"] += payoff
                
                agent.inventory["Products"] = agent.inventory["Coin"]/(self.world.price[-1]+1e-8)

            else:
                # If action > num_labor_hours, this is an error.
                raise ValueError

        total_labor = np.sum(agent_labors)
        total_supply = total_labor * self.world.productivity_per_labor
        year = (self.world.timestep-1)//self.world.period
        self.world.total_products += total_supply
        if year >= len(self.world.nominal_gdp):
            self.world.nominal_gdp.append(0)
            self.world.real_gdp.append(0)
            self.world.unemployment.append(0)
        self.world.nominal_gdp[year] += total_supply*self.world.price[-1]
        self.world.real_gdp[year] += total_supply*self.world.price[0]
        self.world.unemployment[year] += np.sum(np.array(agent_labors)<1)
        if (self.world.timestep % self.world.period == 0) and (self.world.timestep > self.world.period):
            this_year_unemployment = self.world.unemployment[year]
            last_year_unemployment = self.world.unemployment[year-1]
            year_inflation = (this_year_unemployment - last_year_unemployment)/(last_year_unemployment+1e-8)
            self.world.unemployment_rate_inflation.append(year_inflation)
            
            this_year_nominal_gdp = self.world.nominal_gdp[year]
            last_year_nominal_gdp = self.world.nominal_gdp[year-1]
            year_inflation = (this_year_nominal_gdp - last_year_nominal_gdp)/(last_year_nominal_gdp+1e-8)
            self.world.nominal_gdp_inflation.append(year_inflation)
            
            this_year_real_gdp = self.world.real_gdp[year]
            last_year_real_gdp = self.world.real_gdp[year-1]
            year_inflation = (this_year_real_gdp - last_year_real_gdp)/(last_year_real_gdp+1e-8)
            self.world.real_gdp_inflation.append(year_inflation)
        
        if (self.world.timestep % self.world.period == 1) and (self.world.timestep > self.world.period):
            for agent in self.world.agents:
                agent.state["endogenous"]['age'] += 1
        for agent in self.world.agents:
            if agent.get_component_action(self.name):
                agent.state["endogenous"]['job'] = agent.state["endogenous"]['offer']
            else:
                agent.state["endogenous"]['job'] = 'Unemployment'
                agent.state["expected skill"] *= (1 - 0.02)
        

    def generate_observations(self):
        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[str(agent.idx)] = {
                "skill": agent.state["skill"] / self.payment_max_skill_multiplier if self.scale_obs else agent.state["skill"],
                'labor': agent.state["endogenous"]["Labor"],
                'age': agent.state["endogenous"]['age']/60,
            }
        return obs_dict
