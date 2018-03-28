import unittest, numpy

from csb.statistics.pdf.parameterized import Parameter

from isd2 import ArrayParameter
from isd2.pdf.likelihoods import Likelihood
from isd2.model.errormodels import AbstractErrorModel
from isd2.model.forwardmodels import AbstractForwardModel

class MockErrorModel(AbstractErrorModel):

    def __init__(self):

        super(MockErrorModel, self).__init__('StupidErrorModel')

        self._register('ParamB')
        self['ParamB'] = Parameter(4.0, 'ParamB')
        
        self._register_variable('mock_data', differentiable=True)
        self._register_variable('a')

        self.update_var_param_types(mock_data=ArrayParameter, a=Parameter)

        self._set_original_variables()

    def _evaluate_log_prob(self, mock_data, a):

        return a * numpy.sum(mock_data ** 2)

    def _evaluate_gradient(self, mock_data, a):

        return a * 2.0 * mock_data

    def clone(self):

        copy = self.__class__()
        copy.set_fixed_variables_from_pdf(self)

        return copy
    

class MockForwardModel(AbstractForwardModel):

    def __init__(self, parameters=[]):

        super(MockForwardModel, self).__init__('testfwm', parameters)

        self._register_variable('X')
        self._register_variable('b')
        self.update_var_param_types(X=ArrayParameter, b=Parameter)
        self._set_original_variables()

    def _evaluate(self, X, b):

        return b * numpy.array([1.0, 2.0, 3.0])

    def _evaluate_jacobi_matrix(self, X, b):

        return b * numpy.array([[2.0, 1.0, 1.0],
                                [1.0, 2.0, 2.0]])

    def clone(self):

        pass


class NoAutomaticParamsLikelihood(Likelihood):

    def __init__(self, name, forward_model, error_model):

        super(Likelihood, self).__init__(name)
        
        self._forward_model = forward_model
        self._error_model = error_model

        self._set_original_variables()
        

class testLikelihood(unittest.TestCase):

    def setUp(self):

        self.L = Likelihood('testL', MockForwardModel(parameters=[Parameter(2.0, 'ParamA')]), 
                            MockErrorModel())

    def testSetup_parameters(self):

        L = NoAutomaticParamsLikelihood('test', MockForwardModel(parameters=[Parameter(2.0, 'ParamA')]), 
                                        MockErrorModel())

        L._setup_parameters()
        self.assertTrue('ParamA' in L.parameters)
        self.assertTrue(L['ParamA'].value == 2.0)
        L['ParamA'].set(3.0)
        self.assertTrue(L.forward_model['ParamA'].value == 3.0)
        self.assertTrue('ParamB' in L.parameters)
        self.assertTrue(L['ParamB'].value == 4.0)
        L['ParamB'].set(7.0)
        self.assertTrue(L.error_model['ParamB'].value == 7.0)

    def testSplit_variables(self):

        fwm_variables, em_variables = self.L._split_variables({'X': numpy.array([1.0,2.0]), 'a': 5.0, 'b': 2.0})
        self.assertTrue(len(fwm_variables) == 2)
        self.assertTrue(len(em_variables) == 1)
        self.assertTrue('X' in fwm_variables)
        self.assertTrue('b' in fwm_variables)
        self.assertTrue('a' in em_variables)

    def testEvaluate_log_prob(self):

        self.assertTrue(self.L.log_prob(X=numpy.array([1.2, 4.2, 54.5]), a=2.0, b=3.0) == 252.0)

    def testEvaluate_gradient(self):

        a = 2.0
        b = 3.0
        expected = numpy.array([14 * a * b ** 2, 22 * a * b ** 2])
        self.assertTrue(numpy.all(self.L.gradient(X=numpy.array([1.2, 4.2]), a=a, b=b) == expected))

        
if __name__ == '__main__':

    unittest.main()
