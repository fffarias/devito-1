"""
SyncOps are symbolic objects that represent a synchronization operation
on a given object. For example, a SyncOp could represent a locked object
(to avoid race conditions) or a data transfer.
"""

from collections import defaultdict

import sympy

from devito.data import FULL
from devito.tools import Pickable, filter_ordered

__all__ = ['WaitLock', 'WithLock', 'FetchUpdate', 'FetchPrefetch',
           'PrefetchUpdate', 'WaitPrefetch', 'Delete', 'normalize_syncs']


class SyncOp(sympy.Expr, Pickable):

    is_SyncLock = False
    is_SyncData = False

    is_WaitLock = False
    is_WithLock = False

    def __new__(cls, handle):
        obj = sympy.Expr.__new__(cls, handle)
        obj.handle = handle

        return obj

    def __str__(self):
        return "%s<%s>" % (self.__class__.__name__, self.handle)

    __repr__ = __str__

    def __eq__(self, other):
        return type(self) == type(other) and self.args == other.args

    def __hash__(self):
        return hash((type(self), self.args))

    @property
    def imask(self):
        raise NotImplementedError

    # Pickling support
    _pickle_args = ['handle']
    __reduce_ex__ = Pickable.__reduce_ex__


class IMask(object):

    """
    A representation of the data space synchronized by a SyncOp.
    """


class SyncLock(SyncOp):

    is_SyncLock = True

    @property
    def lock(self):
        return self.handle.function

    @property
    def target(self):
        return self.lock.target

    @property
    def size(self):
        return 1


class SyncData(SyncOp):

    is_SyncData = True

    def __new__(cls, dim, size, function, fetch, ifetch, fcond,
                pfetch=None, pcond=None, target=None, tstore=None):
        obj = sympy.Expr.__new__(cls, dim, size, function, fetch, ifetch, fcond,
                                 pfetch, pcond, target, tstore)

        # fetch -> the input Function fetch index, e.g. `time`
        # ifetch -> the input Function initialization index, e.g. `time_m`
        # pfetch -> the input Function prefetch index, e.g. `time+1`
        # tstore -> the target Function store index, e.g. `sb1`

        # fcond -> the input Function fetch condition, e.g. `time_m <= time_M`
        # pcond -> the input Function prefetch condition, e.g. `time + 1 <= time_M`

        obj.dim = dim
        obj.size = size
        obj.function = function
        obj.fetch = fetch
        obj.ifetch = ifetch
        obj.fcond = fcond
        obj.pfetch = pfetch
        obj.pcond = pcond
        obj.target = target
        obj.tstore = tstore

        return obj

    def __str__(self):
        return "%s<%s->%s:%s:%d>" % (self.__class__.__name__, self.function,
                                     self.target, self.dim, self.size)

    __repr__ = __str__

    __hash__ = sympy.Basic.__hash__

    @property
    def dimensions(self):
        return self.function.dimensions

    # Pickling support
    _pickle_args = ['dim', 'size', 'function', 'fetch', 'ifetch', 'fcond']
    _pickle_kwargs = ['pfetch', 'pcond', 'target', 'tstore']
    __reduce_ex__ = Pickable.__reduce_ex__


class WaitLock(SyncLock):
    is_WaitLock = True


class WithLock(SyncLock):

    is_WithLock = True

    def __new__(cls, handle, dspace):
        obj = sympy.Expr.__new__(cls, handle, dspace)
        obj.handle = handle
        obj.dspace = dspace

        return obj

    @property
    def imask(self):
        return [(self.handle.indices[d], self.size) if d.root in self.lock.locked_dimensions else FULL
                for d in self.target.dimensions]


class FetchUpdate(SyncData):

    @property
    def imask(self):
        return [(self.tstore, self.size) if d.root is self.dim.root else FULL
                for d in self.dimensions]


class PrefetchUpdate(SyncData):

    @property
    def imask(self):
        return [(self.tstore, self.size) if d.root is self.dim.root else FULL
                for d in self.dimensions]


class FetchPrefetch(SyncData):

    def iimask(self):
        return [(self.ifetch, self.size) if d.root is self.dim.root else FULL
                for d in self.dimensions]

    def pimask(self):
        return [(self.fetch, self.size) if d.root is self.dim.root else FULL
                for d in self.dimensions]

    def imask(self):
        return [(self.pfetch, self.size) if d.root is self.dim.root else FULL
                for d in self.dimensions]


class WaitPrefetch(SyncData):
    pass


class Delete(SyncData):

    @property
    def imask(self):
        return [(self.fetch, self.size) if d.root is self.dim.root else FULL
                for d in self.dimensions]


def normalize_syncs(*args):
    if not args:
        return
    if len(args) == 1:
        return args[0]

    syncs = defaultdict(list)
    for _dict in args:
        for k, v in _dict.items():
            syncs[k].extend(v)

    syncs = {k: filter_ordered(v) for k, v in syncs.items()}

    for v in syncs.values():
        waitlocks = [i for i in v if i.is_WaitLock]
        withlocks = [i for i in v if i.is_WithLock]

        if waitlocks and withlocks:
            # We do not allow mixing up WaitLock and WithLock ops
            raise ValueError("Incompatible SyncOps")

    return syncs
