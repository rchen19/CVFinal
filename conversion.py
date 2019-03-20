"""
Convert origina SVHN data files to
HDF5 format using Fuel package
"""
import numpy as np
import fuel.converters as con
import os
cur_dir = os.path.dirname(os.path.realpath(__file__))
con.svhn.convert_svhn(which_format=2, directory=cur_dir, output_directory=cur_dir, output_filename=None)