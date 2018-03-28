'''
'''

import unittest

from csb.statistics.pdf.parameterized import Parameter

from isd2.pdf import AbstractISDPDF


class MockParameter(Parameter):

    pass


class MockISDPDF(AbstractISDPDF):

    def __init__(self, name='MockISDPDF'):

        super(MockISDPDF, self).__init__(name=name)

        self._register('ParamA')
        self['ParamA'] = Parameter(2.0, 'ParamA')

        self._register_variable('x')
        self._register_variable('y')

        self.update_var_param_types(x=Parameter, y=Parameter)

        self._set_original_variables()

    def _evaluate_log_prob(self, x, y):

        return -0.5 * self['ParamA'].value * (x ** 2 + y ** 2)

    def clone(self):

        copy = self.__class__()
        copy.set_fixed_variables_from_pdf(self)

        return copy


class testAbstractISDPDF(unittest.TestCase):

    def setUp(self):

        self._mockpdf = MockISDPDF()

    def testFix_variables(self):

        self._mockpdf.fix_variables(y=5.0)
        
        self.assertTrue(len(self._mockpdf.variables) == 1)
        self.assertTrue(list(self._mockpdf.variables)[0]) == 'x'
        self.assertTrue(self._mockpdf['y'].value == 5.0)
        self.assertRaises(ValueError, self._mockpdf.fix_variables, z=2.0)

    def testConditional_factory(self):

        self.setUp()
        cond = self._mockpdf.conditional_factory(x=5.0)
        self.assertTrue('x' in cond.parameters)
        self.assertTrue(cond['x'].value == 5.0)
        self.assertTrue(len(cond.variables) == 1)
        self.assertTrue(list(cond.variables)[0] == 'y')
        self.assertTrue(cond.log_prob(y=2.0) == -29.0)

        cond2 = cond.conditional_factory(y=2.0)
        self.assertTrue('y' in cond2.parameters)
        self.assertTrue(cond2['y'].value == 2.0)
        self.assertTrue(len(cond2.variables) == 0)
        self.assertTrue(cond2.log_prob() == -29.0)

    def testSet_fixed_variables_from_pdf(self):

        pdf1 = MockISDPDF()
        pdf2 = MockISDPDF()

        pdf1.fix_variables(y=2.0)
        pdf2.set_fixed_variables_from_pdf(pdf1)
        self.assertTrue('y' in pdf2.parameters)
        self.assertTrue(pdf2['y'].value == 2.0)

    def testLog_prob(self):

        pdf = MockISDPDF()
        self.assertTrue(pdf.log_prob(x=3, y=2) == -13.0)

    def testUpdate_var_param_types(self):

        pdf = MockISDPDF()
        pdf.update_var_param_types(x=MockParameter)
        self.assertTrue(pdf.var_param_types['x'] == MockParameter)

    def testComplete_variables(self):

        pdf = MockISDPDF()
        pdf.fix_variables(x=7.0)
        variables = {'y': 2.34}
        pdf._complete_variables(variables)
        self.assertTrue('y' in variables)
        self.assertTrue(variables['y'] == 2.34)
        self.assertTrue('x' in variables)
        self.assertTrue(variables['x'] == 7.0)

        
if __name__ == '__main__':

    unittest.main()
