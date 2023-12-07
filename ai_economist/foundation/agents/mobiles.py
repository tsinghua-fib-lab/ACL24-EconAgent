# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from ai_economist.foundation.base.base_agent import BaseAgent, agent_registry


@agent_registry.add
class BasicMobileAgent(BaseAgent):
    """
    A basic mobile agent represents an individual actor in the economic simulation.

    "Mobile" refers to agents of this type being able to move around in the 2D world.
    """

    name = "BasicMobileAgent"
