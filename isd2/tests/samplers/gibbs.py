import unittest, numpy

from isd2.tests.pdf import MockISDPDF
from isd2.samplers import ISDState
from isd2.samplers.gibbs import GibbsSampler
from isd2.samplers.hmc import ISD2FastHMCSampler

class MockSampler(object):

    def __init__(self):

        self.pdf = None
        self._state = 5.0

    @property
    def state(self):
        return self._state
    @state.setter
    def state(self, value):
        self._state = value

    def get_last_draw_stats(self):

        return {'testlastdrawstats{}'.format(self.state): self.state}

    @property
    def sampling_stats(self):

        return {'testsamplingstats{}'.format(self.state): self.state}

    def sample(self):

        if 'y' in self.pdf.parameters:
            return self.state * 2.0 * self.pdf['y'].value
        else:
            return self.state * 2.0 * self.pdf['x'].value


class testGibbsSampler(unittest.TestCase):

    def _create_sampler(self):

        return GibbsSampler(MockISDPDF(), ISDState({'x': 2.0, 'y': 3.0}),
                            {'x': MockSampler(), 'y': MockSampler()})

    def testSetup_conditional_pdfs(self):

        gips = self._create_sampler()

        self.assertTrue(len(gips._conditional_pdfs) == 2)
        self.assertTrue('x' in gips._conditional_pdfs)
        self.assertTrue('y' in gips._conditional_pdfs)
        self.assertTrue(gips._conditional_pdfs['x']['y'].value == 3.0)
        self.assertTrue(gips._conditional_pdfs['y']['x'].value == 2.0)
        self.assertTrue(len(gips._conditional_pdfs['x'].variables) == 1)
        self.assertTrue('y' in gips.subsamplers['x'].pdf.parameters)
        self.assertTrue(gips.subsamplers['x'].pdf['y'].value == 3.0)
        self.assertTrue(len(gips._conditional_pdfs['y'].variables) == 1)
        self.assertTrue('x' in gips.subsamplers['y'].pdf.parameters)
        self.assertTrue(gips.subsamplers['y'].pdf['x'].value == 2.0)

    def testUpdate_conditional_pdf_params(self):

        gips = self._create_sampler()
        
        gips.state.update_variables(x=5.0)
        gips._update_conditional_pdf_params()
        self.assertTrue(gips._conditional_pdfs['y']['x'].value == 5.0)

    def testUpdate_samplers(self):

        gips = self._create_sampler()
        
        new_sampler = MockSampler()
        new_sampler.pdf = MockISDPDF()
        new_sampler.pdf['ParamA'].set(23.0)
        gips.update_samplers(x=new_sampler)

        self.assertTrue(gips.subsamplers['x'].pdf['ParamA'].value == 23.0)

    def testUpdate_subsampler_states(self):

        gips = GibbsSampler(MockISDPDF(), ISDState({'x': 2.0, 'y': numpy.array([3.0])}),
                            {'x': MockSampler(), 
                             'y': ISD2FastHMCSampler(MockISDPDF(), numpy.array([1.0]), 0.1, 12)})

        gips.state.update_variables(x=5.0)
        gips.state.update_variables(y=numpy.array([2.3]))
        gips._update_subsampler_states()

        self.assertTrue(gips.subsamplers['x'].state == 5.0)
        self.assertTrue(gips.subsamplers['y'].state.position == numpy.array([2.3]))

    def testUpdate_state(self):

        gips = self._create_sampler()
        
        gips._update_state(x=34.0)

        self.assertTrue(gips.state.variables['x'] == 34.0)

    def testSample(self):

        gips = self._create_sampler()
        gips._update_state(x=0.5)
        sample = gips.sample()

        self.assertTrue(sample.variables == gips.state.variables)
        self.assertTrue(sample.variables['x'] == 3.0)
        self.assertTrue(sample.variables['y'] == 18.0)

    def testGet_last_draw_stats(self):

        gips = self._create_sampler()
        
        stats = gips.get_last_draw_stats()
        self.assertTrue(len(stats) == 2)
        self.assertTrue('x' in stats)
        self.assertTrue('y' in stats)
        self.assertTrue('testlastdrawstats2.0' in stats['x'])
        self.assertTrue('testlastdrawstats3.0' in stats['y'])

    def testSamplingStats(self):

        gips = self._create_sampler()
        stats = gips.sampling_stats

        self.assertTrue('testsamplingstats2.0' in stats)
        self.assertTrue('testsamplingstats3.0' in stats)
        

if __name__ == '__main__':

    unittest.main()
        
        
