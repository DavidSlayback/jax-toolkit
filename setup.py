"""Install script for setuptools."""

import os
from setuptools import find_namespace_packages
from setuptools import setup

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_version():
  with open('rlax/__init__.py') as fp:
    for line in fp:
      if line.startswith('__version__') and '=' in line:
        version = line[line.find('=') + 1:].strip(' \'"\n')
        if version:
          return version
    raise ValueError('`__version__` not defined in `rlax/__init__.py`')


def _parse_requirements(path):

  with open(os.path.join(_CURRENT_DIR, path)) as f:
    return [
        line.rstrip()
        for line in f
        if not (line.isspace() or line.startswith('#'))
    ]


setup(
    name='jax_toolkit',
    version=_get_version(),
    url='https://github.com/DavidSlayback/jax-toolkit',
    license='GNU 3.0',
    author='David Slayback',
    description=('RL-focused building blocks in JAX'),
    long_description=open(os.path.join(_CURRENT_DIR, 'README.md')).read(),
    long_description_content_type='text/markdown',
    author_email='slayback.d@northeastern.edu',
    keywords='reinforcement-learning python machine learning',
    packages=find_namespace_packages(exclude=['*_test.py']),
    install_requires=_parse_requirements(
        os.path.join(_CURRENT_DIR, 'requirements.txt')),
    tests_require=_parse_requirements(
        os.path.join(_CURRENT_DIR, 'requirements-test.txt')),
    zip_safe=False,  # Required for full installation.
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)