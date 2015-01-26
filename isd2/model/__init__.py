"""
This module contains all "models" occuring in ISD, like the forward models
and the error models.
I'm not quite sure yet whether this module makes any sense. In the ISD2 wiki,
Michael says that "Both should have as many as possible commonalities 
(currently there are differences, one has a name, the other not)."
"""

from abc import abstractmethod

from csb.core import OrderedDict

from isd2.pdf import ParameterNotFoundError, AbstractISDNamedCallable

class AbstractModel(AbstractISDNamedCallable):
    """
    Some copy & pasting from csb.statistics.pdf.AbstractDensity
    involved.
    """

    def __init__(self, name, parameters=[]):

        super(AbstractModel, self).__init__(name)
        
        self._params = OrderedDict()

        for x in parameters:
            self._register(x.name)
            self[x.name] = x

        self.set_params(*parameters)

    def _register(self, name):
        """
        Register a new parameter name.
        """
        if name not in self._params:
            self._params[name] = None

    def __getitem__(self, param):
        
        if param in self._params: 
            return self._params[param]
        else:
            raise ParameterNotFoundError(param)
        
    def __setitem__(self, param, value):
        
        if param in self._params: 
            self._validate(param, value)
            self._params[param] = value
        else:
            raise ParameterNotFoundError(param)

    def _validate(self, param, value):
        """
        Parameter value validation hook.
        @raise ParameterValueError: on failed validation (value not accepted)
        """
        pass

    def set_params(self, *values, **named_params):
        
        for p, v in zip(self.parameters, values):
            self[p].set(v.value)
            
        for p in named_params:
            self[p].set(named_params[p].value)
    
    @property
    def parameters(self):
        """
        Get a list of all distribution parameter names.
        """
        return tuple(self._params)

    def get_params(self):
        return [self._params[name] for name in self.parameters]

    def _complete_variables(self, variables):
        '''
        _complete_variables and _reduce_variables so far only work for classes
        which both inherit from AbstractISDNamedCallable and can hold parameters
        (that is, PDFs and models)
        '''

        variables.update(**{p: self[p].value for p in self.parameters if p in self._original_variables})

    def _reduce_variables(self, **variables):

        for p in self.parameters:
            if p in variables:
                variables.pop(p)
