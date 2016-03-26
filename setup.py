#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
from setuptools import setup, Extension
import numpy as np


setup(name = 'flakes',
      version = "0.0.1",
      author = "Daniel Beck",
      author_email = "daniel.e.beck@gmail.com",
      description = ("Structural kernels in Tensorflow"),
      keywords = "machine-learning kernels tensorflow",
      url = "none yet",
      ext_modules = [],
      packages = ["flakes"],
      package_dir={'flakes': 'flakes'},
      py_modules = ['flakes.__init__'],
      test_suite = 'tests',
      install_requires=['numpy>=1.9', 'scipy>=0.16', 'tensorflow>=0.7'],
      )
