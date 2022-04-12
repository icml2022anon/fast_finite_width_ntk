"""Setup the package with pip."""


import os
import setuptools


# https://packaging.python.org/guides/making-a-pypi-friendly-readme/
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()


INSTALL_REQUIRES = [
    'jax>=0.2.25',
    'tensorflow==2.7.0',
    'torch==1.10',
    'onnx2keras==0.2.24',
    'onnx==1.11.0'
]

# Then install `tf2jax` without dependencies:
# pip install git+https://github.com/deepmind/tf2jax.git --no-deps


def _get_version() -> str:
  """Returns the package version.
  Adapted from:
  https://github.com/deepmind/dm-haiku/blob/d4807e77b0b03c41467e24a247bed9d1897d336c/setup.py#L22
  Returns:
    Version number.
  """
  path = 'fast_finite_width_ntk/__init__.py'
  version = '__version__'
  with open(path) as fp:
    for line in fp:
      if line.startswith(version):
        g = {}
        exec(line, g)  # pylint: disable=exec-used
        return g[version]  # pytype: disable=key-error
    raise ValueError(f'`{version}` not defined in `{path}`.')


setuptools.setup(
    name='fast-finite-width-ntk',
    version=_get_version(),
    license='Apache 2.0',
    author='Anonymous',
    author_email='icml2022anon@gmail.com',
    install_requires=INSTALL_REQUIRES,
    url='https://github.com/icml2022anon/fast_finite_width_ntk',
    download_url='https://github.com/icml2022anon/fast_finite_width_ntk',
    project_urls={
        'Source Code': 'https://github.com/icml2022anon/fast_finite_width_ntk',
        'Paper': 'https://cmt3.research.microsoft.com/ICML2022/Submission/Summary/6626',
    },
    packages=setuptools.find_packages(exclude=('presentation',)),
    long_description=long_description,
    long_description_content_type='text/markdown',
    description='Fast Finite Width Neural Tangent Kernel',
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: MacOS',
        'Operating System :: POSIX :: Linux',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Development Status :: 4 - Beta',
    ])
