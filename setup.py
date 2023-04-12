import os
import codecs

from setuptools import setup


pkg_name = 'pysparta'
version = '1.0.0'
author = 'Jose A Ruiz-Arias'
author_email = 'jararias@uma.es'
url = ''
description = (
    'Solar PArameterization for the Radiative Transfer of the Atmosphere'
)
keywords = ['solar radiation', 'solar energy', 'simulation']
classifiers = [
    "Natural Language :: English",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3",
    "Development Status :: 4 - Beta",
]

with codecs.open(f'{pkg_name}/_version.py', 'w', 'utf-8') as f:
    f.write(f'__version__ = "{version}"')

def read_content(fname):
    CURDIR = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(CURDIR, fname), 'rb', 'utf-8') as f:
        return f.read()

setup(
    name=pkg_name,
    version=version,
    author=author,
    author_email=author_email,
    url=url,
    description=description,
    long_description=read_content('README.md'),
    long_description_content_type='text/markdown',
    keywords=keywords,
    classifers=classifiers,
    packages=[
        pkg_name,
    ],
    package_dir={
        pkg_name: pkg_name
    },
    # scripts=[
    #     f'scripts/{pkg_name}-latency',
    #     f'scripts/{pkg_name}-benchmark-random'
    # ],
    python_requires=">=3.8",
    install_requires=['numpy', 'numexpr', 'loguru'],
)

if os.path.exists(f'{pkg_name}/_version.py'):
    os.remove(f'{pkg_name}/_version.py')
