# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 17:06:29 2016

@author: Kirill Konevets
email:   kkonevets@gmail.com

CONTENT:
    
"""

from distutils.core import setup

from Cython.Build import cythonize

setup(
    ext_modules=cythonize("./ai/cmatching.pyx")
)
