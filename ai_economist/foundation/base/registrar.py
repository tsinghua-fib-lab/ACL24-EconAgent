# Copyright (c) 2020, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


class Registry:
    """Utility for registering sets of similar classes and looking them up by name.

    Registries provide a simple API for getting classes used to build environment
    instances. Their main purpose is to organize such "building block" classes (i.e.
    Components, Scenarios, Agents) for easy reference as well as to ensure that all
    classes within a particular registry inherit from the same Base Class.

    Args:
        base_class (class): The class that all entries in the registry must be a
            subclass of.

    Example:
        class BaseClass:
            pass

        registry = Registry(BaseClass)

        @registry.add
        class ExampleSubclassA(BaseClass):
            name = "ExampleA"
            pass

        @registry.add
        class ExampleSubclassB(BaseClass):
            name = "ExampleB"
            pass

        print(registry.entries)
        # ["ExampleA", "ExampleB"]

        assert registry.has("ExampleA")
        assert registry.get("ExampleB") is ExampleSubclassB
    """

    def __init__(self, base_class=None):
        self.base_class = base_class
        self._entries = []
        self._lookup = dict()

    def add(self, cls):
        """Add cls to this registry.

        Args:
            cls: The class to add to this registry. Must be a subclass of
                self.base_class.

        Returns:
            cls (to allow decoration with @registry.add)

        See Registry class docstring for example.
        """
        assert "." not in cls.name
        if self.base_class:
            assert issubclass(cls, self.base_class)
        self._lookup[cls.name.lower()] = cls
        if cls.name not in self._entries:
            self._entries.append(cls.name)
        return cls

    def get(self, cls_name):
        """Return registered class with name cls_name.

        Args:
            cls_name (str): Name of the registered class to get.

        Returns:
            Registered class cls, where cls.name matches cls_name (ignoring casing).

        See Registry class docstring for example.
        """
        if cls_name.lower() not in self._lookup:
            raise KeyError('"{}" is not a name of a registered class'.format(cls_name))
        return self._lookup[cls_name.lower()]

    def has(self, cls_name):
        """Return True if a class with name cls_name is registered.

        Args:
            cls_name (str): Name of class to check.

        See Registry class docstring for example.
        """
        return cls_name.lower() in self._lookup

    @property
    def entries(self):
        """Names of classes in this registry.

        Returns:
            A list of strings corresponding to the names of classes registered in
                this registry object.

        See Registry class docstring for example.
        """
        return sorted(list(self._entries))
