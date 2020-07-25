"""
binf top level module. Submodules provide interfaces for posterior components
"""
from abc import ABCMeta, abstractmethod

import numpy
import numpy as np, sys

from csb.statistics.pdf.parameterized import AbstractParameter


class AbstractBinfNamedCallable(object, metaclass=ABCMeta):

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
        # see docstring of register_variables for info what the following
        # line is about
        self.variable_names = {}

    def _set_original_variables(self):
        """
        Stores the original set of named variables this object takes
        """
        self._original_variables.update(self.variables)

    def _register_variable(self, name, kind=None, differentiable=False):
        """
        Registers a variable so that later on it can be fixed or
        checked whether it has been passed to the __call__ method

        :param name: name of the new variable
        :type name: str

        :param kind: the "kind" of the variable. This is a weird concept:
                     if two instances of this class describe the same type
                     of function. say, a normal distribution with the precision
                     as a variable, then you will need to give a different 
                     variable name to each of the two precisions. But both
                     variables are of kind "precision". All these hoops are
                     required so that you can just instantiate two normal
                     distribution objects with different precision variable
                     names as arguments, all the while the implementation of
                     the normal distribution is agnostic of the actual variable
                     name and only requires the variable kind.
                     In this case, you would want to choose "precision" as kind,
                     but, for example, "first_precision" and "second_precision"
                     as variable names.
        :type kind: str

        :param differentiable: True for variables you might at one point
                               want to take the gradient w.r.t.
                               This is probably deprecated.
        :type differentiable: bool

        """
        if type(name) != str or  not (kind is None or type(kind) == str):
            raise ValueError(('Variable name / kind must be a string, '
                              'not ' + type(name)))
        elif name in self._variables:
            raise ValueError('Variable name \"' + name + '\" must be unique')
        else:
            variable_kind = kind or name
            self.variable_names[variable_kind] = name
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
        complete_mapped_vars = self._complete_and_map_variables(**variables)
        result = self._evaluate(**complete_mapped_vars)

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
        if len(variables.keys() & set(self._differentiable_variables)) == 0:
            msg = 'Function cannot be differentiated w.r.t.' + \
                  'any of the variables '+list(variables.keys())
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
        complete_mapped_vars = self._complete_and_map_variables(**variables)
        result = self._evaluate_gradient(**complete_mapped_vars)
        
        return result
    
    @abstractmethod
    def _complete_variables(self, variables):
        """
        If needed, updates the dictionary of variables passed to this object
        when calling evaluate() with values for fixed variables
        """
        pass

    def _complete_and_map_variables(self, **variables):
        """
        This completes the set of variables with possibly fixed variable
        values and returns the mapped set of variables, in which the
        variable names are eliminated.
        """
        self._complete_variables(variables)
        mapped_vars = {kind: variables[name]
                       for kind, name in self.variable_names.items()}

        return mapped_vars
    
    def _get_variables_intersection(self, test_variables):
        """
        Returns the intersection of the variables stored in the argument
        with the set of currently registered variables
        """
        return {k: v for k, v in list(test_variables.items()) if k in self.variables}

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


class ArrayParameter(AbstractParameter):
            
    def _validate(self, value):
        try:
            return numpy.array(value)
        except(TypeError, ValueError):
            raise ParameterValueError(self.name, value)
