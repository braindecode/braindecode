from setuptools import setup, find_packages  # Always prefer setuptools over distutils
from codecs import open  # To use a consistent encoding
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the relevant file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()


# This will add __version__ to version dict
version = {}
with open(path.join(here, 'braindecode/version.py'), encoding='utf-8') as (
        version_file):
    exec(version_file.read(), version)

setup(
    name='Braindecode',

    version=version['__version__'],

    description='A deep learning toolbox to decode raw time-domain EEG.',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/robintibor/braindecode',

    # Author details
    author='Robin Tibor Schirrmeister',
    author_email='robintibor@gmail.com',

    # Choose your license
    license='BSD 3-Clause',

    install_requires=['mne',  'numpy', 'pandas', 'scipy',
                      'resampy', 'matplotlib', 'h5py',],
    #tests_require = [...]

    # See https://PyPI.python.org/PyPI?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        'Topic :: Software Development :: Build Tools',

        "Topic :: Scientific/Engineering :: Artificial Intelligence",

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='eeg deep-learning brain-state-decoding',

    packages=find_packages(),
    include_package_data=False,
    zip_safe=False,
)
