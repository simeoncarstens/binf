# First steps with binf #

## What is binf about?
binf is a collection of classes I use for Bayesian inference problems: probability distributions (likelihood, posterior, priors) and forward- and error models and a Gibbs- and HMC sampler. I use this code in my research on Bayesian chromatin structure determination ([Carstens et al., PLOS Comput. Biol. 2016](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005292)). The code has limited functionality (namely, it does what I need it to do), but maybe you'll find it useful. 

## Setting up binf
binf has two dependencies (if I remember correctly): numpy ([download](https://pypi.python.org/pypi/numpy)) and the Computational Structural Biology Toolbox (CSB, [download](https://github.com/csb-toolbox/CSB)). Install these and you should be able to install binf by typing

        $ python setup.py install
    
possibly with the `--user` option, if you don't have administrator privileges.

## Tests / documentation
For most classes, there are unit tests to make sure things work as they're supposed to. You can run the tests by typing

        $ cd binf
        $ python run_test.py

The code is not yet commented / documented. I tried to pay attention to some coding practices, though, so hopefully it is not too bad to read / use / adapt. But there is a fully worked-out example, which infers the coefficients of a polynomial model to some data points and plots the results. You can run it by typing

        $ cd binf
        $ python example_script.py

(requires matplotlib). This example and the imports from the `example/` subfolder should give you an idea of how the code works.

## Contact
If you have questions, don't hesitate to drop me a message.

## License
binf is open source and distributed under the OSI-approved MIT license.

    Copyright (c) 2018 Simeon Carstens

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
