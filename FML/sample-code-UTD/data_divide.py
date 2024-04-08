import random
import numpy as np
import pandas as pd
import os
from os.path import exists,join
import shutil

TEST_DATA_PATH = '/code/MMFederated/FML/sample-code-UTD/UTD-data/split-6-2/train/label-27-5-percent/unlabel'

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

def divide_label_and_unlabel_data(label_percent):
    """
    根据label_percent将现有的训练集中labeled和unlabeled数据比例重新进行划分
    """
    BASIC_PATH = '/code/MMFederated/FML/sample-code-UTD/UTD-data/split-6-2/train'
    # step1: make the folder
    os.makedirs(join(BASIC_PATH,f'label-27-{label_percent}-percent'),exist_ok=True)
    os.makedirs(join(BASIC_PATH,f'label-27-{label_percent}-percent','label'),exist_ok=True)
    os.makedirs(join(BASIC_PATH,f'label-27-{label_percent}-percent','unlabel'),exist_ok=True)
    os.makedirs(join(BASIC_PATH,f'label-27-{label_percent}-percent','label','inertial'),exist_ok=True)
    os.makedirs(join(BASIC_PATH,f'label-27-{label_percent}-percent','label','skeleton'),exist_ok=True)
    os.makedirs(join(BASIC_PATH,f'label-27-{label_percent}-percent','unlabel','inertial'),exist_ok=True)
    os.makedirs(join(BASIC_PATH,f'label-27-{label_percent}-percent','unlabel','skeleton'),exist_ok=True)


    inertial_files = os.listdir(join(BASIC_PATH,'total','inertial'))
    skeleton_files = os.listdir(join(BASIC_PATH,'total','skeleton'))
    assert len(inertial_files) == len(skeleton_files)
    labeled_num_per_action = round(len(inertial_files) * label_percent / 2700)
    for i in range(1,28):
        inertial_subclass_list = [item for item in inertial_files if item.split('_')[0] == f'a{i}']
        skeleton_subclass_list = [item for item in skeleton_files if item.split('_')[0] == f'a{i}']
        assert len(inertial_subclass_list) == len(skeleton_subclass_list)
        labeled_index = random.sample(range(len(inertial_subclass_list)),labeled_num_per_action)
        unlabeled_index = list(set(range(len(inertial_subclass_list)))-set(labeled_index))
        labeled_inertial_subclass_list = [inertial_subclass_list[i] for i in labeled_index]
        labeled_skeleton_subclass_list = [skeleton_subclass_list[i] for i in labeled_index]
        unlabeled_inertial_subclass_list = [inertial_subclass_list[i] for i in unlabeled_index]
        unlabeled_skeleton_subclass_list = [skeleton_subclass_list[i] for i in unlabeled_index]
        for item in labeled_inertial_subclass_list:
            file_name = join(BASIC_PATH,'total','inertial',item)
            shutil.copy(file_name,join(BASIC_PATH,f'label-27-{label_percent}-percent','label','inertial'))
        for item in labeled_skeleton_subclass_list:
            file_name = join(BASIC_PATH,'total','skeleton',item)
            shutil.copy(file_name,join(BASIC_PATH,f'label-27-{label_percent}-percent','label','skeleton'))

        for item in unlabeled_inertial_subclass_list:
            file_name = join(BASIC_PATH,'total','inertial',item)
            shutil.copy(file_name,join(BASIC_PATH,f'label-27-{label_percent}-percent','unlabel','inertial'))
        for item in unlabeled_skeleton_subclass_list:
            file_name = join(BASIC_PATH,'total','skeleton',item)
            shutil.copy(file_name,join(BASIC_PATH,f'label-27-{label_percent}-percent','unlabel','skeleton'))


if __name__ == '__main__':
    # inertial_files = os.listdir(join(TEST_DATA_PATH,'inertial'))
    # skeleton_files = os.listdir(join(TEST_DATA_PATH,'skeleton'))
    # # divide_by_user(inertial_files,skeleton_files)
    # # divide_by_class(inertial_files,skeleton_files)
    # print('len(inertial_files):', len(inertial_files))
    # print('len(skeleton_files):', len(skeleton_files))

    divide_label_and_unlabel_data(20)