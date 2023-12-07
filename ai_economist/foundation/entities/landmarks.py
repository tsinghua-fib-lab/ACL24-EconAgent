# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from ai_economist.foundation.base.registrar import Registry
from ai_economist.foundation.entities.resources import resource_registry


class Landmark:
    """Base class for Landmark entity classes.

    Landmark classes describe the entities that exist exclusively in the environment
    world. In other words, they represent entities that should not be included in an
    agent's inventory and are only observable through observations from the
    spatial world.

    Landmark classes describe the following properties:
         ownable: If each instance of the landmark belongs to an agent. For example, a
            "House" is ownable and belongs to the agent that constructs it whereas
            "Water" is not ownable.
        solid: If the landmark creates a physical barrier to movement (that is,
            if agents are prevented from occupying cells with the landmark).
            Importantly, if the landmark is ownable, the agent that owns a given
            landmark can occupy its cell even if the landmark is solid.
    """

    name = None
    color = None  # array of RGB values [0 - 1]
    ownable = None
    solid = True  # Solid = Cannot be passed through
    # (unless it is owned by the agent trying to pass through)

    def __init__(self):
        assert self.name is not None
        assert self.color is not None
        assert self.ownable is not None

        # No agent can pass through this landmark
        self.blocking = self.solid and not self.ownable

        # Only the agent that owns this landmark can pass through it
        self.private = self.solid and self.ownable

        # This landmark does not belong to any agent and it does not inhibit movement
        self.public = not self.solid and not self.ownable


landmark_registry = Registry(Landmark)

# Registering each collectible resource's source block
# allows treating source blocks in a specific way
for resource_name in resource_registry.entries:
    resource = resource_registry.get(resource_name)
    if not resource.collectible:
        continue

    @landmark_registry.add
    class SourceBlock(Landmark):
        """Special Landmark for generating resources. Not ownable. Not solid."""

        name = "{}SourceBlock".format(resource.name)
        color = np.array(resource.color)
        ownable = False
        solid = False


@landmark_registry.add
class House(Landmark):
    """House landmark. Ownable. Solid."""

    name = "House"
    color = np.array([220, 20, 220]) / 255.0
    ownable = True
    solid = True


@landmark_registry.add
class Water(Landmark):
    """Water Landmark. Not ownable. Solid."""

    name = "Water"
    color = np.array([50, 50, 250]) / 255.0
    ownable = False
    solid = True
