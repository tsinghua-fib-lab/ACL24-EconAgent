# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from ai_economist.foundation.base.base_agent import BaseAgent, agent_registry


@agent_registry.add
class BasicPlanner(BaseAgent):
    """
    A basic planner agent represents a social planner that sets macroeconomic policy.

    Unlike the "mobile" agent, the planner does not represent an embodied agent in
    the world environment. BasicPlanner modifies the BaseAgent class to remove
    location as part of the agent state.

    Also unlike the "mobile" agent, the planner agent is expected to be unique --
    that is, there should only be 1 planner. For this reason, BasicPlanner ignores
    the idx argument during construction and always sets its agent index as "p".
    """

    name = "BasicPlanner"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.state["loc"]

        # Overwrite any specified index so that this one is always indexed as 'p'
        # (make a separate class of planner if you want there to be multiple planners
        # in a game)
        self._idx = "p"

    @property
    def loc(self):
        """
        Planner agents do not occupy any location.
        """
        raise AttributeError("BasicPlanner agents do not occupy a location.")
