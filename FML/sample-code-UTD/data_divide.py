import numpy as np
import pandas as pd
import os
from os.path import exists,join
import shutil

TEST_DATA_PATH = '/code/MMFederated/FML/sample-code-UTD/UTD-data/data'

SPLIT_BY_USER_PATH = '/code/MMFederated/FML/sample-code-UTD/UTD-data/split_by_user'
SPLIT_BY_CLASS_PATH = '/code/MMFederated/FML/sample-code-UTD/UTD-data/split_by_class'

def traverse_data_dict(data_dict):
    if isinstance(data_dict, dict):
        for key, value in data_dict.items():
            traverse_data_dict(value)
    elif isinstance(data_dict, list):
        print('len(data_dict):', len(data_dict))

def divide_by_user(inertial_files,skeleton_files):
    data_dict = {}
    for inertial_file in inertial_files:
        user_id = int((inertial_file.split('_')[1]).replace('s',''))
        if user_id not in data_dict:
            data_dict[user_id] = {}
        if 'inertial' not in data_dict[user_id]:
            data_dict[user_id]['inertial'] = []
        data_dict[user_id]['inertial'].append(inertial_file)
    
    for skeleton_file in skeleton_files:
        user_id = int((skeleton_file.split('_')[1]).replace('s',''))
        if user_id not in data_dict:
            data_dict[user_id] = {}
        if 'skeleton' not in data_dict[user_id]:
            data_dict[user_id]['skeleton'] = []
        data_dict[user_id]['skeleton'].append(skeleton_file)

    for use_id,sub_dict in data_dict.items():
        for tensor_type,val in sub_dict.items():
            for item in val:
                file_name = join(TEST_DATA_PATH,tensor_type,item)
                if not exists(join(SPLIT_BY_USER_PATH,str(use_id),tensor_type)):
                    os.mkdir(join(SPLIT_BY_USER_PATH,str(use_id),tensor_type))
                shutil.copy(file_name,join(SPLIT_BY_USER_PATH,str(use_id),tensor_type))

def divide_by_class(inertial_files,skeleton_files):
    data_dict = {}
    for inertial_file in inertial_files:
        class_id = int((inertial_file.split('_')[0]).replace('a',''))
        if class_id not in data_dict:
            data_dict[class_id] = {}
        if 'inertial' not in data_dict[class_id]:
            data_dict[class_id]['inertial'] = []
        data_dict[class_id]['inertial'].append(inertial_file)

    for skeleton_file in skeleton_files:
        class_id = int((skeleton_file.split('_')[0]).replace('a',''))
        if class_id not in data_dict:
            data_dict[class_id] = {}
        if 'skeleton' not in data_dict[class_id]:
            data_dict[class_id]['skeleton'] = []
        data_dict[class_id]['skeleton'].append(skeleton_file)

    for key in data_dict.keys():
        if not exists(join(SPLIT_BY_CLASS_PATH,str(key))):
            os.mkdir(join(SPLIT_BY_CLASS_PATH,str(key)))

    for class_id,sub_dict in data_dict.items():
        for tensor_type,val in sub_dict.items():
            for item in val:
                file_name = join(TEST_DATA_PATH,tensor_type,item)
                if not exists(join(SPLIT_BY_CLASS_PATH,str(class_id),tensor_type)):
                    os.mkdir(join(SPLIT_BY_CLASS_PATH,str(class_id),tensor_type))
                shutil.copy(file_name,join(SPLIT_BY_CLASS_PATH,str(class_id),tensor_type))

if __name__ == '__main__':
    inertial_files = os.listdir(join(TEST_DATA_PATH,'inertial'))
    skeleton_files = os.listdir(join(TEST_DATA_PATH,'skeleton'))

    # divide_by_user(inertial_files,skeleton_files)
    divide_by_class(inertial_files,skeleton_files)

    print('len(inertial_files):', len(inertial_files))
    print('len(skeleton_files):', len(skeleton_files))