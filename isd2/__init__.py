"""
Future ISD stuff goes here. Currently, the major focus is to redesign the
Universe and access to atoms, molecules etc. Also Posterior and ISDSampler
will be redesigned at some point.
"""
__version__ = '2.0.0'

from abc import ABCMeta, abstractmethod

import numpy
import numpy as np, sys

from csb.statistics.pdf.parameterized import AbstractParameter

sys.setrecursionlimit(int(1e6))

## TODO: will we need this at all?
class isdobject(object):

    import numpy as np

DEBUG = True


class AbstractISDNamedCallable(object):

    __metaclass__ = ABCMeta

    def __init__(self, name):

        self._name = name
        self._variables = set()
        self._differentiable_variables = set()
        self._var_param_types = {}

    def _set_original_variables(self):

        self._original_variables = self.variables.copy()

    def _register_variable(self, name, differentiable=False):

        if type(name) != str:
            raise ValueError('Variable name must be a string, not ' + type(name))
        elif name in self._variables:
            raise ValueError('Variable name \"' + name + '\" must be unique')
        else:
            self._variables.add(name)
            if differentiable:
                self._differentiable_variables.add(name)
  
    def _delete_variable(self, name):

        if name in self._variables:
            self._variables.remove(name)
            if name in self.differentiable_variables:
                self._differentiable_variables.remove(name)
        else:
            raise ValueError('\"' + name + '\": unknown variable name')

    @property
    def variables(self):
        return self._variables

    @property
    def differentiable_variables(self):
        return self._differentiable_variables

    @property
    def name(self):
        return self._name

    def __call__(self, **variables):        

        self._complete_variables(variables)
        result = self._evaluate(**variables)

        return result

    @abstractmethod
    def _evaluate(self, **variables):
        pass

    def _check_differentiability(self, **variables):

        if len(variables.viewkeys() & set(self._differentiable_variables)) == 0:
            msg = 'Function cannot be differentiated w.r.t. any of the variables '+variables.keys()
            raise ValueError(msg)

    def _evaluate_gradient(self, **variables):

        raise NotImplementedError

    def gradient(self, **variables):

        self._complete_variables(variables)
        result = self._evaluate_gradient(**variables)
        
        return result
    
    @abstractmethod
    def _complete_variables(self, variables):
        pass
    
    def _get_variables_intersection(self, test_variables):

        return {k: v for k, v in test_variables.items() if k in self.variables}

    @property
    def var_param_types(self):
        '''
        Empty by default, but for a conditional PDF to be built,
        one has to add AbstractParameter subclasses suiting the variables.
        '''
        return self._var_param_types.copy()
    def update_var_param_types(self, **values):
        self._var_param_types.update(**values)


class ArrayParameter(AbstractParameter):
            
    def _validate(self, value):
        try:
            return numpy.array(value)
        except(TypeError, ValueError):
            raise ParameterValueError(self.name, value)


    

import functools, hashlib


## copyed & pasted from https://gist.github.com/dpo/1222577
## if we keep this, we should ask for permission
class memoize(object):
    """
    Decorator class used to cache the most recent value of a function or method
    based on the signature of its arguments. If any single argument changes,
    the function or method is evaluated afresh.
    """

    def __init__(self, callable):
        self._callable = callable
        self._callable_is_method = False
        self.value = None # Cached value or derivative.
        self._args_signatures = {}
        return


    def __get_signature(self, x):
        # Return signature of argument.
        # The signature is the value of the argument or the sha1 digest if the
        # argument is a numpy array.
        # Subclass to implement other digests.
        if isinstance(x, np.ndarray):
            _x = x.view(np.uint8)
            return hashlib.sha1(_x).hexdigest()
        return x


    def __call__(self, *args, **kwargs):
        # The callable will be called if any single argument is new or changed.

        callable = self._callable
        evaluate = False

        # If we're memoizing a class method, the first argument will be 'self'
        # and need not be memoized.
        firstarg = 1 if self._callable_is_method else 0

        # Get signature of all arguments.
        nargs = callable.func_code.co_argcount # Non-keyword arguments.
        argnames = callable.func_code.co_varnames[firstarg:nargs]
        argvals = args[firstarg:]

        for (argname,argval) in zip(argnames,argvals) + kwargs.items():

            _arg_signature = self.__get_signature(argval)

            try:
                cached_arg_sig = self._args_signatures[argname]
                if cached_arg_sig != _arg_signature:
                    self._args_signatures[argname] = _arg_signature
                    evaluate = True

            except KeyError:
                self._args_signatures[argname] = _arg_signature
                evaluate = True

        # If all arguments are unchanged, return cached value.
        if evaluate:
            self.value = callable(*args, **kwargs)

        return self.value

    def __get__(self, obj, objtype):
        "Support instance methods."
        self._callable_is_method = True
        return functools.partial(self.__call__, obj)


    def __repr__(self):
        "Return the wrapped function or method's docstring."
        return self.method.__doc__

