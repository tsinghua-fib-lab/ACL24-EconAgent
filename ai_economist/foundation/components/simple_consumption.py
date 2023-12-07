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
class SimpleConsumption(BaseComponent):

    name = "SimpleConsumption"
    required_entities = ["Coin"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        mask_first_step=True,
        consumption_rate_step=0.02,
        max_price_inflation=0.1,
        max_wage_inflation=0.05,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        # Define consumption rate step
        self.consumption_rate_step = consumption_rate_step
        self.n_actions = int(1.0/consumption_rate_step)

        assert isinstance(mask_first_step, bool)
        self.mask_first_step = mask_first_step

        self.is_first_step = True
        self.common_mask_on = {
            agent.idx: np.ones((self.n_actions,)) for agent in self.world.agents
        }
        self.common_mask_off = {
            agent.idx: np.zeros((self.n_actions,)) for agent in self.world.agents
        }

        self.max_price_inflation = max_price_inflation
        self.max_wage_inflation = max_wage_inflation

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
    
    def get_additional_state_fields(self, agent_cls_name):
        return {}
    
    def additional_reset_steps(self):
        self.is_first_step = True

    def component_step(self):
        
        last_total_products = self.world.total_products
        total_demand = 0
        for agent in self.world.get_random_order_agents():

            action = agent.get_component_action(self.name)

            if action == 0:  # NO-OP.
                # Agent is not interacting with this component.
                continue

            if 1 <= action <= self.n_actions:  # set reopening phase

                consumption_rate = action*self.consumption_rate_step
                agent.state["endogenous"]["Consumption Rate"] = consumption_rate

                consumption = agent.total_endowment("Coin") * consumption_rate
                demand = consumption/(self.world.price[-1]+1e-8)
                total_demand += demand
                if self.world.total_products >= demand:
                    self.world.total_products -= demand
                else:
                    consumption = self.world.total_products*self.world.price[-1]
                    self.world.total_products = 0
                
                # Actually consume
                agent.consumption["Coin"] = consumption
                agent.state["inventory"]["Coin"] -= consumption
                agent.consumption["Products"] = agent.consumption["Coin"]/(self.world.price[-1]+1e-8)

            else:
                raise ValueError

        max_change_rate = (total_demand - last_total_products)/(max(total_demand, last_total_products)+1e-8)
        # Wage Change
        if self.world.enable_skill_change:
            
            for agent in self.world.agents:
                
                agent.state["skill"] = max(agent.state["skill"]*(1 + np.random.uniform(0, max_change_rate*self.max_wage_inflation)), 1)

            self.world.wage.append(np.mean([agent.state["skill"] for agent in self.world.agents]))
            
        if (self.world.timestep % self.world.period == 0) and (self.world.timestep > self.world.period):
            this_year_wage = np.mean(self.world.wage[-self.world.period:])
            last_year_wage = np.mean(self.world.wage[-2*self.world.period:-self.world.period])
            year_inflation = (this_year_wage - last_year_wage)/last_year_wage
            self.world.wage_inflation.append(year_inflation)
        
        # Price Change
        if self.world.enable_price_change:
            this_inflation = np.random.uniform(0, max_change_rate*self.max_price_inflation)
            
            self.world.price.append(max(self.world.price[-1]*(1 + this_inflation), 1))
        
        if (self.world.timestep % self.world.period == 0) and (self.world.timestep > self.world.period):
            this_year_price = np.mean(self.world.price[-self.world.period:])
            last_year_price = np.mean(self.world.price[-2*self.world.period:-self.world.period])
            year_inflation = (this_year_price - last_year_price)/last_year_price
            self.world.inflation.append(year_inflation)
            
        # self.set_offer()

    def generate_observations(self):
        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[str(agent.idx)] = {
                "Consumption Rate": agent.state["endogenous"]["Consumption Rate"],
                'price': self.world.price[-1]/self.world.price[0]
            }
        return obs_dict
