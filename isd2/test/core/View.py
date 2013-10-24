if __name__ == '__main__':

    from isd2.core.Viewable import Viewable
    from isd2.core.View import View
    
    class A(Viewable):
        def __init__(self, state):
            super(A,self).__init__()
            self.state = state

        @property
        def state(self):
            return self._state

        @state.setter
        def state(self, state):
            self._state = state
            self.changed()

    class AView(View):

        def __init__(self,a):
            super(AView,self).__init__(a)
            self._nchanges = 0
            
        def update(self):
            self._nchanges += 1
            if self._View__values is None:
                self._View__values = -1
            self._View__values += 1
            
    a = A(1)
    v = AView(a)
    print v.get()

    a.state = 10
    print v.get()

    a.state = 'asdfaf'
    print v.get()
        
        
