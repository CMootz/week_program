#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io

from os.path import dirname
from os.path import join

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    file_path = join(dirname(__file__), *names)
    with io.open(file_path, encoding=kwargs.get('encoding', 'utf8')) as fh:
        return fh.read()


setup(
    name='week_program',
    version='0.1.0',
    license='MIT license',
    description='Week Program',
    long_description=read('README.md'),
    author='Christian Mootz',
    author_email='christian.mootz@mootz-automation.de',
    url='https://github.com/CMootz/week_program',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    zip_safe=False,
)
