import numpy as np
import pandas as pd
import os
from os.path import exists,join

TEST_DATA_PATH = '/code/MMFederated/FML/UTD-data/split-6-2/test'

if __name__ == '__main__':
    inertial_files = os.listdir(join(TEST_DATA_PATH,'inertial'))
    skeleton_files = os.listdir(join(TEST_DATA_PATH,'skeleton'))
    print('len(inertial_files):', len(inertial_files))
    print('len(skeleton_files):', len(skeleton_files))