import os
import sys
import imp

from distutils.core import setup, Extension

try:
    __doc__ = open('README.txt').read()
except IOError:
    pass

__file__ = './'
ROOT            = 'isd2'
LOCATION        = os.path.abspath(os.path.dirname(__file__))
JUNK            = ['CVS']

NAME            = "isd2"
VERSION         = imp.load_source('____isd', os.path.join(LOCATION, ROOT, '__init__.py')).__version__
AUTHOR          = "Michael Habeck et al."
EMAIL           = "michael.habeck@tuebingen.mpg.de"
URL             = "http://www.eb.tuebingen.mpg.de/research/research-groups/michael-habeck.html"
SUMMARY         = "Inferential Structure Determination"
DESCRIPTION     = __doc__
LICENSE         = 'MIT'
REQUIRES        = ['numpy', 'scipy']

def extension():
    return 
    try:
        import numpy
    except:
        raise ImportError('Numpy dependency not found')

    isdmodule = Extension('isd._isd',
                          define_macros = [('MAJOR_VERSION', '2'),
                                           ('MINOR_VERSION', '0'),
                                           ('PY_ARRAY_UNIQUE_SYMBOL','ISD')],
                          include_dirs = [numpy.get_include(), './isd/c/include'],
                          ##                     libraries = ['tcl83'],
                          # library_dirs = ['/usr/local/lib'],
                          extra_compile_args = [],
                          sources = ['./isd/c/_isdmodule.c',
                                     './isd/c/universe/atom.c',
                                     './isd/c/polymer/bond.c',
                                     './isd/c/polymer/rigidgroup.c',
                                     './isd/c/polymer/rotation.c',
                                     './isd/c/polymer/polymer.c',
                                     './isd/c/universe/universe.c',
                                     './isd/c/misc/find_pairs.cpp',
                                     './isd/c/misc/utils.c',
                                     './isd/c/misc/mathutils.c',
                                     './isd/c/misc/distancematrix.c',
                                     './isd/c/misc/interactiongrid.c',
                                     './isd/c/forcefield/prolsq.c',
                                     './isd/c/forcefield/nblist.c',
                                     './isd/c/prior/boltzmann.c',
                                     './isd/c/prior/tsallis.c',
                                     './isd/c/prior/ramachandran.c',
                                     './isd/c/theory/ispa.c',
                                     './isd/c/theory/karpluscurve.c',
                                     './isd/c/theory/torsionrestraint.c',
                                     './isd/c/theory/distancerestraint.c',
                                     './isd/c/theory/saupe.c',
                                     './isd/c/errormodel/errormodel.c',
                                     './isd/c/errormodel/lognormal.c',
                                     './isd/c/errormodel/normal.c',
                                     './isd/c/errormodel/lognormal_marginal.c',
                                     './isd/c/errormodel/vonmises.c',
                                     './isd/c/errormodel/lowerupper.c',
                                     './isd/c/data/datum.c',
                                     './isd/c/data/noe.c',
                                     './isd/c/data/jcoupling.c',
                                     './isd/c/data/torsionangle.c',
                                     './isd/c/data/rdc.c',
                                     './isd/c/data/dataset.c',
                                     './isd/c/forcefield/hbonds.c', 
                                     './isd/c/forcefield/solvent.c', 
                                     './isd/c/forcefield/prolsqhb.c', 
                                     './isd/c/forcefield/rosettahb.c', 
                                     './isd/c/forcefield/nbgrid.c', 
                                     './isd/c/prior/rotamers.c', 
                                     './isd/c/prior/ramachandran2.c', 
                                     './isd/c/theory/alignment.c', 
                                     './isd/c/errormodel/normal_individual.c', 
                                     './isd/c/errormodel/upper_individual.c', 
                                     './isd/c/errormodel/weightedlognormal.c', 
                                     './isd/c/errormodel/weightednormal.c', 
                                     './isd/c/theory/mixturespectrum.c', 
                                     './isd/c/theory/noesy.c', 
                                     './isd/c/theory/noesy3D.c', 
                                     './isd/c/theory/noesy3D2.c', 
                                     './isd/c/theory/map3D.c', 
                                     './isd/c/theory/spin.c', 
                                     './isd/c/theory/grid.c', 
                                     './isd/c/theory/map.c', 
                                     './isd/c/theory/ispa2.c', 
                                     './isd/c/theory/distancerestraint2.c', 
                                     './isd/c/theory/order_parameter.c',
                                     './isd/c/theory/emmap.c',                                     
                                     './isd/c/forcefield/eef1.c', 
                                     './isd/c/forcefield/rosetta.c', 
                                     './isd/c/errormodel/noncentralmaxwell.c', 
                                     './isd/c/errormodel/vonmises_weighted.c',
                                     './isd/c/data/peak.c',  
                                     './isd/c/theory/freq_grid.c', 
                                     './isd/c/theory/spectrum.c', 
                                     './isd/c/theory/contact.c', 
                                     './isd/c/errormodel/lognormal_goodbad.c', 
                                     './isd/c/errormodel/lognormal_individual.c', 
                                     './isd/c/errormodel/lognormal_erfc.c', 
                                     './isd/c/errormodel/poisson.c', 
                                     './isd/c/errormodel/logstudent.c', 
                                     './isd/c/errormodel/loglaplace.c', 
                                     './isd/c/forcefield/harmonicff.c', 
                                     ])


    return isdmodule

def sourcetree(root='isd2', junk=('CVS','isd32','isd64')):
    """
    Since distutils requires to HARD CODE the entire package hierarchy here,
    we need this function to load the source tree structure dynamically.
    
    @param root: name of the root package; this is 'isd'. Must be relative!
    @type root: str
    @param junk: skip those directories
    @type junk: tuple
    
    @return: a list of "package" names
    @rtype: list  
    """
    junk = set(junk)
    items = []

    curdir = os.path.abspath(os.curdir)
    cwd = os.path.dirname(__file__) or '.'
    os.chdir(cwd)
    
    if root.startswith(os.path.sep):
        raise ValueError('root must be a relative path')
    elif not os.path.isdir(os.path.join('.', root)):
        raise ValueError('root package "{0}" not found in {1}.'.format(root, cwd))        
    
    for entry in os.walk(root):
            
        directory = entry[0]
        parts = set(directory.split(os.path.sep))
            
        init = os.path.join(directory, '__init__.py')
        # take all packages: directories with __init__, except if a junk 
        # directory is found at any level in the tree
        if os.path.isfile(init) and junk.isdisjoint(parts):
            items.append(directory)

    os.chdir(curdir)
        
    return items

def datatree(package, dataroot, junk=('.svn','.cvs'), mask='*.*'):
    """
    Since distutils will crash if the data root folder contains any subfolders,
    we need this function to retrieve the data tree.

    @param package: root "package", containing a data folder. This is a 
                    relative path, e.g. "csb/test"
    @type package: str
    @param dataroot: name of the data root directory for C{package},
                     relative to C{package}
    @type dataroot: str
    @param junk: skip those directories
    @type junk: tuple
    
    @return: a list of all glob patterns with all subdirectories of the data
             root, including the root itself. The paths  returned are relative
             to C{package}  
    @rtype: list      
    """
    junk = set(junk)
    items = []

    curdir = os.path.abspath(os.curdir)
    cwd = os.path.dirname(__file__) or '.'
    os.chdir(cwd)
    
    if package.startswith(os.path.sep):
        raise ValueError('package must be a relative path')
    elif not os.path.isdir(os.path.join('.', package)):
        raise ValueError('package "{0}" not found in {1}.'.format(package, cwd))        

    os.chdir(package)
        
    for entry in os.walk(dataroot):
        
        directory = entry[0]
        parts = set(directory.split(os.path.sep))
        
        # take all directories, except if a junk dir is found at any level in the tree
        if junk.isdisjoint(parts):
            item = os.path.join(directory, mask)
            items.append(item)

    os.chdir(curdir)
        
    return items

def build():
    
    toppar = os.path.join('isd', 'toppar')
    
    return setup(\
                name=NAME,
                packages=sourcetree(ROOT, JUNK),
                package_dir = {'isd2': './isd2'},
        ## package_data={'isd2': ['./toppar/[a-z]*']},
                version=VERSION,
                author=AUTHOR,
                author_email=EMAIL,
                url=URL,
                description=SUMMARY,
                long_description=DESCRIPTION,
                license=LICENSE,
                requires=REQUIRES,
                ext_modules=[], #[extension()],
                classifiers=(\
                    'Development Status :: 5 - Production/Stable',
                    'Intended Audience :: Developers',
                    'Intended Audience :: Science/Research',
                    'License :: OSI Approved :: MIT License',
                    'Operating System :: OS Independent',
                    'Programming Language :: Python',
                    'Programming Language :: Python :: 2.6',
                    'Programming Language :: Python :: 2.7',
                    'Programming Language :: Python :: 3.1',
                    'Programming Language :: Python :: 3.2',                    
                    'Topic :: Scientific/Engineering',
                    'Topic :: Scientific/Engineering :: Bio-Informatics',
                    'Topic :: Scientific/Engineering :: Mathematics',
                    'Topic :: Scientific/Engineering :: Physics',
                    'Topic :: Software Development :: Libraries'
            )
        
    )



if __name__ == '__main__':
    
    build()
