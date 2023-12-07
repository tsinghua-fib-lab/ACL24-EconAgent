# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import numpy as np

from ai_economist.foundation.base.registrar import Registry


class Resource:
    """Base class for Resource entity classes.

    Resource classes describe entities that can be a part of an agent's inventory.

    Resources can also be a part of the world as collectible entities: for each
    Resource class with Resource.collectible=True, a complementary
    ResourceSourceBlock Landmark class will be created in landmarks.py. For each
    collectible resource in the environment, the world map will include a resource
    source block channel (representing landmarks where collectible resources are
    generated) and a resource channel (representing locations where collectible
    resources have generated).
    """

    name = None
    color = None  # array of RGB values [0 - 1]
    collectible = None  # Is this something that exists in the world?
    # (versus something that can only be owned)

    def __init__(self):
        assert self.name is not None
        assert self.color is not None
        assert self.collectible is not None


resource_registry = Registry(Resource)


@resource_registry.add
class Wood(Resource):
    """Wood resource. collectible."""

    name = "Wood"
    color = np.array([107, 143, 113]) / 255.0
    collectible = True


@resource_registry.add
class Stone(Resource):
    """Stone resource. collectible."""

    name = "Stone"
    color = np.array([241, 233, 219]) / 255.0
    collectible = True


@resource_registry.add
class Coin(Resource):
    """Coin resource. Included in all environments by default. Not collectible."""

    name = "Coin"
    color = np.array([229, 211, 82]) / 255.0
    collectible = False
    
@resource_registry.add
class Products(Resource):
    """Coin resource. Included in all environments by default. Not collectible."""

    name = "Products"
    color = np.array([211, 82, 229]) / 255.0
    collectible = False
