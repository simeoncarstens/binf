import unittest

loader = unittest.TestLoader()
tests = loader.discover('./binf/tests/', pattern='*.py')
testRunner = unittest.runner.TextTestRunner()
testRunner.run(tests)
