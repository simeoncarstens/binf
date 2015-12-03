"""
Core classes
"""
class Node(object):
    """
    A tree
    """
    def __init__(self, node):

        object.__init__(self)

        self._info = node
        self._parent = None
        self._children = set()

    def __str__(self):

        return 'Node({})'.format(self._info)

    @property
    def info(self):
        return self._info

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return list(self._children)

    def is_root(self):
        return self.parent is None
    
    def as_dict(self):

        dict = {self: self.children}

        for child in self.children:
            dict.update(child.as_dict())

        return dict

    def from_dict(self, dict):

        for child in dict[self]:

            child.link(self)
            child.from_dict(dict)

    def get_siblings(self):
        
        if self.is_root(): return []

        siblings = list(self.parent.children)
        siblings.remove(self)

        return siblings

    def get_ancestors(self, n_generations = 1):

        if self.is_root() or n_generations < 1: return []

        ancestors = [self.parent]
        ancestors += ancestors[-1].get_ancestors(n_generations - 1)

        return ancestors

    def get_descendants(self, n_generations = None):
        """
        Get all children that are hanging off this parent.
        """
        descendants = [self]

        if type(n_generations) is int:
            n_generations -= 1
            if n_generations < 0:
                return descendants
            
        for child in self.children:
            descendants += child.get_descendants(n_generations)

        return descendants

    def is_child(self, group):

        return group in self.children

    def has_children(self):

        return len(self.children) > 0

    def link(self, parent):

        if parent is None: return

        if self._parent is not None:
            msg = 'Node already linked'
            raise ValueError(msg)

        self._parent = parent

        if self in parent._children:
            print 'Warning: already linked: {0} {1}'.format(self, parent)
        else:
            parent._children.add(self)

    def unlink(self):

        if not self.is_root():

            self._parent._children.remove(self)
            self._parent = None

    def trace_back(self):

        trace = [self]

        generation = self

        while not generation.is_root():

            generation = generation.parent
            trace.insert(0, generation)

        return trace

