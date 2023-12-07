# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

from ai_economist.foundation.base.registrar import Registry


class Endogenous:
    """Base class for endogenous entity classes.

    Endogenous entities are those that, conceptually, describe the internal state
    of an agent. This provides a convenient way to separate physical entities (which
    may exist in the world, be exchanged among agents, or are otherwise in principal
    observable by others) from endogenous entities (such as the amount of labor
    effort an agent has experienced).

    Endogenous entities are registered in the "endogenous" portion of an agent's
    state and should only be observable by the agent itself.
    """

    name = None

    def __init__(self):
        assert self.name is not None


endogenous_registry = Registry(Endogenous)


@endogenous_registry.add
class Labor(Endogenous):
    """Labor accumulated through working. Included in all environments by default."""

    name = "Labor"
