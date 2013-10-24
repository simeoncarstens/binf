"""
Implements a Viewable object

A Viewable object will notify all its views whenever its own
state changed and invalidate them.
"""

class Viewable(object):
    """
    An object that can be viewed by one or multiple views which
    will be notified / invalidated if the state of the object
    changes.
    """
    def __init__(self):        
        self._views = []

    def add_view(self, view):
        """
        Add a view
        """
        if not view in self._views: self._views.append(view)

    def del_view(self, view):
        """
        Delete a view
        """
        if not view in self._views:
            raise ValueError("{0} not viewed by {1}".format(self, view))
        self._views.remove(view)

    def changed(self):
        """
        Should be called by a viewable object once its state has changed
        and all views watching the object should be notified.
        """
        for view in self._views: view.invalidate()

        
