"""
This module provides general functionality and classes
related to probability density functions and their parameters.
"""

from abc import ABCMeta, abstractmethod

from isd2 import AbstractISDNamedCallable

from csb.numeric import exp
from csb.statistics.pdf.parameterized import ParameterizedDensity


class ParameterNotFoundError(AttributeError):

    pass


class AbstractISDPDF(ParameterizedDensity, AbstractISDNamedCallable):

    __metaclass__ = ABCMeta

    def __init__(self, name='', **args):
        r"""
        Defines the interface for probability density functions,
        combining the CSB PDF classes with the interface for functions
        taking typed and named variables

        :param name: name for this object
        :type name: str

        :param \**args: arguments required for instantiation
        """
        ParameterizedDensity.__init__(self)
        AbstractISDNamedCallable.__init__(self, name)

        self._var_param_types = {}

    @property
    def estimator(self):
        raise NotImplementedError
    @estimator.setter
    def estimator(self, strategy):
        pass

    def estimate(self, data):
        raise NotImplementedError

    def conditional_factory(self, **fixed_vars):
        """
        Makes a copy of this object in which one or several variables
        are set to specific values (on which the new PDF is 'conditioned'
        on)

        The 'conditioning' must not be taken literally, as no prior
        probabilities are taken into account. But that doesn't bother us,
        as we're going to MCMC sampling anyways

        :param \**variables: keyword arguments containg name / value pairs
                             for variables on which this PDF should be conditioned
                             on
        
        :returns: PDF object conditioned on the given values
        :rtype: :class:`.AbstractISDPDF
        """

        result = self.clone()
        result.fix_variables(**self._get_variables_intersection(fixed_vars))

        return result            
    
    @abstractmethod
    def _evaluate_log_prob(self, **variables):
        r"""
        In this method, the actual evaluation of the log-probability
        takes place

        The variables argument holds values for both fixed and unfixed
        variables; the implementation in this method thus does not depend
        on whether the set of originally passed variables equals the set
        of original variables

        :param \**variables: list of variable name / value pairs
        """
        pass

    def _evaluate(self, **variables):

        return exp(self.log_prob(**variables))

    def log_prob(self, **variables):
        r"""
        Evaluates the log-probability of the PDF represented by this
        object

        :param \**variables: list of variable name / value pairs

        :returns: log-probability
        :rtype: float
        """
        self._complete_variables(variables)
        result = self._evaluate_log_prob(**variables)

        return result
    
    def gradient(self, **variables):

        self._complete_variables(variables)
        result = self._evaluate_gradient(**variables)

        return result

    def fix_variables(self, **fixed_vars):
        """
        Sets ('fixes') specific variables to values given as keyword
        arguments by removing these variables from the list of registered
        variables and registering corresponding parameters to this object.
        """
        for v in fixed_vars:
            if v in self.variables:
                self._delete_variable(v)
                self._register(v)
                if v in self.var_param_types:
                    self[v] = self.var_param_types[v](fixed_vars[v], v)
                else:
                    msg = 'Parameter type for variable "{}" not defined'.format(v)
                    raise ValueError(msg)
            else:
                msg = '{} is not a variable of {}'.format(self.__repr__(), v)
                raise ValueError(msg)

    @abstractmethod
    def clone(self):
        """
        Returns an exact copy (same parameters / variables...) of this object

        :returns: a copy of this object
        :rtype: :class:`.AbstractISDPDF`
        """
        pass

    def set_fixed_variables_from_pdf(self, pdf):
        """
        Retrieves fixed variables from another PDF object and fixes the same
        variables in this object accordingly

        :param pdf: PDF object to retrieve variables to fix from
        :type pdf: :class:`.AbstractISDPDF`        
        """
        variables = {p: pdf[p].value for p in pdf.parameters if not p in self.parameters}
        self.fix_variables(**self._get_variables_intersection(variables))

    def _complete_variables(self, variables):
        '''
        _complete_variables and _reduce_variables so far only work for classes
        which both inherit from AbstractISDNamedCallable and can hold parameters
        (that is, PDFs and models)
        '''

        variables.update(**{p: self[p].value for p in self.parameters if p in self._original_variables})


class TestHO(AbstractISDPDF):

    def __init__(self, k=1.0, x0=0.0, name='TestHO'):

        from csb.statistics.pdf.parameterized import Parameter
        from hicisd2.hicisd2lib import ArrayParameter
        
        super(TestHO, self).__init__(name=name)

        self._register('k')
        self._register('x0')
        self['k'] = Parameter(k, name='k')
        self['x0'] = Parameter(x0, name='x0')
        self._register_variable('x', differentiable=True)

        self._set_original_variables()
        self.update_var_param_types(x=ArrayParameter)

    def _evaluate_log_prob(self, x):

        import numpy
        
        return -0.5 * self['k'].value * numpy.sum((x - self['x0'].value) ** 2)

    def _evaluate_gradient(self, x):

        import numpy
        
        return self['k'].value * (x - self['x0'].value)
