"""
Future Binf stuff goes here. Currently, the major focus is to redesign the
Universe and access to atoms, molecules etc. Also Posterior and BinfSampler
will be redesigned at some point.
"""
__version__ = '2.0.0'

from abc import ABCMeta, abstractmethod

import numpy
import numpy as np, sys

from csb.statistics.pdf.parameterized import AbstractParameter


class AbstractBinfNamedCallable(object):

    __metaclass__ = ABCMeta

    def __init__(self, name):
        """
        Class defining the interface functions taking named and typed
        arguments ('variables')

        :param name: some unique name for this object
        :type name: str
        """
        self._name = name
        self._variables = set()
        self._differentiable_variables = set()
        self._var_param_types = {}
        self._original_variables = set()

    def _set_original_variables(self):
        """
        Stores the original set of named variables this object takes
        """
        self._original_variables.update(self.variables)

    def _register_variable(self, name, differentiable=False):
        """
        Registers a variable so that later on it can be fixed or
        checked whether it has been passed to the __call__ method

        :param name: name of the new variable
        :type name: str

        :param differentiable: True for variables you might at one point
                               want to take the gradient w.r.t.
                               This is probably deprecated.
        :type differentiable: bool

        """
        if type(name) != str:
            raise ValueError('Variable name must be a string, not ' + type(name))
        elif name in self._variables:
            raise ValueError('Variable name \"' + name + '\" must be unique')
        else:
            self._variables.add(name)
            if differentiable:
                self._differentiable_variables.add(name)
  
    def _delete_variable(self, name):
        """
        Removes a variable from the set of registed variables

        :param name: name of the variable to be removed
        :type name: str
        """
        if name in self._variables:
            self._variables.remove(name)
            if name in self.differentiable_variables:
                self._differentiable_variables.remove(name)
        else:
            raise ValueError('\"' + name + '\": unknown variable name')

    @property
    def variables(self):
        """
        Returns the set of currently registed variables

        :returns: set of currently registered variables
        :rtype: set
        """
        return self._variables

    @property
    def differentiable_variables(self):
        """
        Returns the set of currently registered variables this object
        implements the gradient w.r.t

        :returns: set of variables this object can be differentiated w.r.t.
        :rtype: set
        """
        return self._differentiable_variables

    @property
    def name(self):
        """
        Returns the name of this object
        """
        return self._name

    def __call__(self, **variables):
        """
        Evaluates the function described by this object

        :returns: function value
        :rtype: unknown
        """

        if len(variables) != len(self.variables):
            msg = 'Function called with {}' + \
                  'arguments instead of {}!'.format(len(variables),                                                                       len(self.variables))
            raise ValueError(msg)
        self._complete_variables(variables)
        result = self._evaluate(**variables)

        return result

    @abstractmethod
    def _evaluate(self, **variables):
        r"""
        In this method, the actual function evaluation takes place

        The variables argument holds values for both fixed and unfixed
        variables; the implementation in this method thus does not depend
        on whether the set of originally passed variables equals the set
        of original variables

        :param \**variables: list of variable name / value pairs
        """
        pass

    def _check_differentiability(self, **variables):
        """
        Checks whether this object can be differentiated w.r.t. specific
        variables 
        """
        if len(variables.viewkeys() & set(self._differentiable_variables)) == 0:
            msg = 'Function cannot be differentiated w.r.t.' + \
                  'any of the variables '+variables.keys()
            raise ValueError(msg)

    def _evaluate_gradient(self, **variables):
        r"""
        In this method, the actual gradient evaluation takes place

        The variables argument holds values for both fixed and unfixed
        variables; the implementation in this method thus does not depend
        on whether the set of originally passed variables equals the set
        of original variables

        :param \**variables: list of variable name / value pairs
        """

        raise NotImplementedError

    def gradient(self, **variables):
        r"""
        This function will be called from outside to evaluate the gradient
        of the function (or a related one, e.g., log-probability) represented
        by this object

        :param \**variables: list of variable name / value pairs

        :returns: gradient of the function represented by this object
        :rtype: :class:`numpy.ndarray`
        """

        if len(variables) != len(self.variables):
            msg = 'Function called with {}' + \
                  'arguments instead of {}!'.format(len(variables),                                                                       len(self.variables))
            raise ValueError(msg)            
        self._complete_variables(variables)
        result = self._evaluate_gradient(**variables)
        
        return result
    
    @abstractmethod
    def _complete_variables(self, variables):
        """
        If needed, updates the dictionary of variables passed to this object
        when calling evaluate() with values for fixed variables
        """
        pass
    
    def _get_variables_intersection(self, test_variables):
        """
        Returns the intersection of the variables stored in the argument
        with the set of currently registered variables
        """
        return {k: v for k, v in test_variables.items() if k in self.variables}

    @property
    def var_param_types(self):
        '''
        Empty by default, but for a conditional PDF to be built,
        one has to add AbstractParameter subclasses suiting the variables.
        '''
        return self._var_param_types.copy()
    def update_var_param_types(self, **values):
        """
        Updates the dictionary holding the types of registered variables
        """
        self._var_param_types.update(**values)

    @abstractmethod
    def fix_variables(self, **fixed_vars):
        """
        Sets ('fixes') specific variables to values given as keyword
        arguments
        """
        pass


class ArrayParameter(AbstractParameter):
            
    def _validate(self, value):
        try:
            return numpy.array(value)
        except(TypeError, ValueError):
            raise ParameterValueError(self.name, value)
