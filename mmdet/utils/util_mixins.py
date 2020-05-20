"""
This module defines the :class:`NiceRepr` mixin class, which defines a
``__repr__`` and ``__str__`` method that only depend on a custom ``__nice__``
method, which you must define. This means you only have to overload one
function instead of two.  Furthermore, if the object defines a ``__len__``
method, then the ``__nice__`` method defaults to something sensible, otherwise
it is treated as abstract and raises ``NotImplementedError``.

To use simply have your object inherit from :class:`NiceRepr`
(multi-inheritance should be ok).

This code was copied from the ubelt library: https://github.com/Erotemic/ubelt

Example:
    >>> # Objects that define __nice__ have a default __str__ and __repr__
    >>> class Student(NiceRepr):
    ...    def __init__(self, name):
    ...        self.name = name
    ...    def __nice__(self):
    ...        return self.name
    >>> s1 = Student('Alice')
    >>> s2 = Student('Bob')
    >>> print(f's1 = {s1}')
    >>> print(f's2 = {s2}')
    s1 = <Student(Alice)>
    s2 = <Student(Bob)>

Example:
    >>> # Objects that define __len__ have a default __nice__
    >>> class Group(NiceRepr):
    ...    def __init__(self, data):
    ...        self.data = data
    ...    def __len__(self):
    ...        return len(self.data)
    >>> g = Group([1, 2, 3])
    >>> print(f'g = {g}')
    g = <Group(3)>

"""
import warnings


class NiceRepr(object):
    """
    Inherit from this class and define ``__nice__`` to "nicely" print your
    objects.

    Defines ``__str__`` and ``__repr__`` in terms of ``__nice__`` function
    Classes that inherit from :class:`NiceRepr` should redefine ``__nice__``.
    If the inheriting class has a ``__len__``, method then the default
    ``__nice__`` method will return its length.

    Example:
        >>> class Foo(NiceRepr):
        ...    def __nice__(self):
        ...        return 'info'
        >>> foo = Foo()
        >>> assert str(foo) == '<Foo(info)>'
        >>> assert repr(foo).startswith('<Foo(info) at ')

    Example:
        >>> class Bar(NiceRepr):
        ...    pass
        >>> bar = Bar()
        >>> import pytest
        >>> with pytest.warns(None) as record:
        >>>     assert 'object at' in str(bar)
        >>>     assert 'object at' in repr(bar)

    Example:
        >>> class Baz(NiceRepr):
        ...    def __len__(self):
        ...        return 5
        >>> baz = Baz()
        >>> assert str(baz) == '<Baz(5)>'
    """

    def __nice__(self):
        if hasattr(self, '__len__'):
            # It is a common pattern for objects to use __len__ in __nice__
            # As a convenience we define a default __nice__ for these objects
            return str(len(self))
        else:
            # In all other cases force the subclass to overload __nice__
            raise NotImplementedError(
                f'Define the __nice__ method for {self.__class__!r}')

    def __repr__(self):
        try:
            nice = self.__nice__()
            classname = self.__class__.__name__
            return f'<{classname}({nice}) at {hex(id(self))}>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

    def __str__(self):
        try:
            classname = self.__class__.__name__
            nice = self.__nice__()
            return f'<{classname}({nice})>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)
