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
class SimpleSaving(BaseComponent):

    name = "SimpleSaving"
    required_entities = ["Coin"]
    agent_subclasses = ["BasicMobileAgent"]

    def __init__(
        self,
        *base_component_args,
        mask_first_step=True,
        saving_rate=0.02,
        scale_obs=True,
        **base_component_kwargs
    ):
        super().__init__(*base_component_args, **base_component_kwargs)

        assert isinstance(mask_first_step, bool)
        self.mask_first_step = mask_first_step
        self.n_actions = None

        self.is_first_step = True
        self.common_mask_on = {}
        self.common_mask_off = {}
        
        self.scale_obs = scale_obs
        
    def get_additional_state_fields(self, agent_cls_name):
        return {}

    def additional_reset_steps(self):
        self.is_first_step = True

    def get_n_actions(self, agent_cls_name):
        if agent_cls_name == "BasicMobileAgent":
            return self.n_actions
        return None

    def generate_masks(self, completions=0):
        return {}


    def component_step(self):

        if (self.world.timestep % self.world.period == 0):
            for agent in self.world.get_random_order_agents():
                
                saving = agent.total_endowment("Coin")

                payoff = self.world.interest_rate[-1]*saving
                agent.state["inventory"]["Coin"] += payoff
                
            if self.world.timestep > self.world.period:
                natural_interest_rate = 0.01
                target_inflation = 0.02
                if len(self.world.inflation) > 0:
                    natural_unemployment_rate = 0.04
                    inflation_coeff, unemployment_coeff = 0.5, 0.5
                    tao = 1
                    avg_inflation = np.mean(self.world.inflation[-tao:])
                    year = (self.world.timestep-1)//self.world.period
                    avg_unemployment_rate = np.mean(np.array(self.world.unemployment[max(year-tao+1, 0):year+1])/self.world.period/self.world.n_agents)
                    interest_rate = natural_interest_rate + target_inflation + inflation_coeff * (avg_inflation - target_inflation) + unemployment_coeff * (natural_unemployment_rate - avg_unemployment_rate)
                else:
                    interest_rate = natural_interest_rate + target_inflation
                    
                self.world.interest_rate.append(max(interest_rate, 0))
        
        
            
    def generate_observations(self):
        obs_dict = dict()
        for agent in self.world.agents:
            obs_dict[str(agent.idx)] = {
                "Saving Return": self.world.interest_rate[-1],
                'wealth': agent.inventory["Coin"] / (self.world.timestep+1) if self.scale_obs else agent.inventory["Coin"]
            }
        return obs_dict
