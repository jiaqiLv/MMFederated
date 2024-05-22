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

def divide_label_and_unlabel_data(label_rate):
    """
    根据label_percent将现有的训练集中labeled和unlabeled数据比例重新进行划分
    """
    if label_rate == 5:
        train_folder = 'label-27-5-percent'
        label_file_name = ['a1_s6_t3', 'a2_s5_t4', 'a3_s3_t4', 'a4_s2_t2', 'a5_s4_t4', 'a6_s3_t2', 'a7_s5_t4', 'a8_s2_t3', 'a9_s3_t1', 
		'a10_s4_t3', 'a11_s6_t2', 'a12_s4_t4', 'a13_s2_t3', 'a14_s5_t1', 'a15_s4_t2', 'a16_s2_t4', 'a17_s6_t1', 'a18_s6_t2', 
		'a19_s5_t4', 'a20_s6_t1', 'a21_s2_t3', 'a22_s5_t1', 'a23_s6_t3', 'a24_s5_t4', 'a25_s3_t2', 'a26_s1_t1', 'a27_s3_t2']
    elif label_rate == 10:
        train_folder = 'label-54-10-percent'
        label_file_name = ['a1_s6_t3', 'a1_s5_t4', 'a2_s3_t1', 'a2_s2_t3', 'a3_s4_t3', 'a3_s3_t3', 'a4_s5_t4', 'a4_s2_t3', 'a5_s3_t4', 
		'a5_s4_t3', 'a6_s6_t3', 'a6_s4_t2', 'a7_s2_t3', 'a7_s5_t2', 'a8_s4_t3', 'a8_s2_t2', 'a9_s6_t1', 'a9_s5_t1', 'a10_s5_t3', 'a10_s6_t3', 
		'a11_s2_t2', 'a11_s5_t1', 'a12_s6_t4', 'a12_s5_t4', 'a13_s3_t4', 'a13_s1_t4', 'a14_s3_t4', 'a14_s4_t4', 'a15_s6_t3', 'a15_s6_t2', 'a16_s3_t3',
		 'a16_s6_t2', 'a17_s2_t4', 'a17_s5_t4', 'a18_s4_t4', 'a18_s1_t3', 'a19_s5_t4', 'a19_s3_t3', 'a20_s5_t2', 'a20_s5_t3', 'a21_s1_t4', 'a21_s3_t4',
		 'a22_s6_t4', 'a22_s2_t1', 'a23_s2_t3', 'a23_s6_t2', 'a24_s2_t3', 'a24_s4_t4', 'a25_s2_t1', 'a25_s6_t3', 'a26_s5_t1', 'a26_s3_t1', 'a27_s1_t4', 'a27_s2_t2']
    elif label_rate == 15:
        train_folder = 'label-81-15-percent'
        label_file_name = ['a1_s6_t4', 'a1_s5_t3', 'a1_s3_t2', 'a2_s2_t3', 'a2_s4_t2', 'a2_s3_t4', 'a3_s5_t4', 'a3_s2_t4', 'a3_s3_t3', 'a4_s4_t4', 
		'a4_s6_t3', 'a4_s4_t2', 'a5_s2_t3', 'a5_s5_t4', 'a5_s4_t4', 'a6_s2_t4', 'a6_s6_t1', 'a6_s6_t3', 'a7_s5_t2', 'a7_s6_t3', 'a7_s2_t4', 'a8_s5_t1', 
		'a8_s6_t3', 'a8_s4_t1', 'a9_s3_t1', 'a9_s1_t4', 'a9_s3_t2', 'a10_s4_t4', 'a10_s6_t1', 'a10_s5_t1', 'a11_s3_t3', 'a11_s6_t1', 'a11_s2_t3', 
		'a12_s5_t4', 'a12_s4_t3', 'a12_s1_t3', 'a13_s5_t1', 'a13_s3_t3', 'a13_s5_t3', 'a14_s5_t3', 'a14_s1_t2', 'a14_s3_t2', 'a15_s6_t4', 'a15_s2_t1', 
		'a15_s1_t1', 'a16_s6_t4', 'a16_s2_t1', 'a16_s4_t1', 'a17_s2_t1', 'a17_s6_t4', 'a17_s5_t4', 'a18_s3_t1', 'a18_s1_t2', 'a18_s2_t1', 'a19_s4_t2', 
		'a19_s6_t1', 'a19_s1_t3', 'a20_s4_t2', 'a20_s5_t1', 'a20_s4_t3', 'a21_s5_t1', 'a21_s4_t1', 'a21_s6_t4', 'a22_s4_t1', 'a22_s4_t2', 'a22_s3_t3', 
		'a23_s4_t4', 'a23_s3_t4', 'a23_s4_t1', 'a24_s2_t3', 'a24_s2_t4', 'a24_s2_t1', 'a25_s4_t3', 'a25_s4_t4', 'a25_s3_t2', 'a26_s1_t2', 'a26_s5_t3', 
		'a26_s6_t2', 'a27_s6_t1', 'a27_s6_t2', 'a27_s5_t2']
    elif label_rate == 20:
        train_folder = 'label-108-20-percent'
        label_file_name = ['a1_s2_t3', 'a1_s1_t1', 'a1_s3_t1', 'a1_s1_t2', 'a2_s1_t1', 'a2_s3_t4', 'a2_s6_t1', 'a2_s5_t3', 
		'a3_s5_t4', 'a3_s2_t2', 'a3_s4_t4', 'a3_s3_t2', 'a4_s2_t1', 'a4_s1_t2', 'a4_s3_t1', 'a4_s6_t2', 'a5_s5_t1', 'a5_s5_t4', 
		'a5_s6_t4', 'a5_s2_t2', 'a6_s2_t1', 'a6_s4_t2', 'a6_s5_t1', 'a6_s6_t1', 'a7_s6_t1', 'a7_s1_t4', 'a7_s4_t1', 'a7_s5_t1', 
		'a8_s4_t4', 'a8_s2_t3', 'a8_s3_t4', 'a8_s1_t2', 'a9_s6_t4', 'a9_s6_t3', 'a9_s4_t4', 'a9_s2_t3', 'a10_s6_t2', 'a10_s4_t1', 
		'a10_s6_t1', 'a10_s6_t4', 'a11_s4_t4', 'a11_s3_t4', 'a11_s4_t2', 'a11_s3_t3', 'a12_s1_t4', 'a12_s2_t3', 'a12_s5_t4', 'a12_s1_t3', 
		'a13_s1_t3', 'a13_s4_t3', 'a13_s2_t2', 'a13_s4_t4', 'a14_s3_t4', 'a14_s3_t1', 'a14_s6_t4', 'a14_s2_t4', 'a15_s3_t3', 'a15_s2_t1', 
		'a15_s4_t4', 'a15_s3_t1', 'a16_s3_t4', 'a16_s5_t3', 'a16_s2_t1', 'a16_s4_t1', 'a17_s6_t3', 'a17_s1_t3', 'a17_s2_t3', 'a17_s2_t4', 
		'a18_s5_t1', 'a18_s4_t1', 'a18_s2_t4', 'a18_s2_t1', 'a19_s2_t4', 'a19_s3_t1', 'a19_s1_t4', 'a19_s5_t4', 'a20_s6_t4', 'a20_s6_t3', 
		'a20_s5_t4', 'a20_s6_t1', 'a21_s1_t3', 'a21_s2_t2', 'a21_s6_t4', 'a21_s5_t4', 'a22_s3_t2', 'a22_s6_t3', 'a22_s4_t1', 'a22_s5_t1', 
		'a23_s2_t3', 'a23_s2_t2', 'a23_s3_t4', 'a23_s1_t3', 'a24_s3_t1', 'a24_s6_t2', 'a24_s2_t4', 'a24_s1_t3', 'a25_s1_t1', 'a25_s2_t4', 
		'a25_s5_t4', 'a25_s3_t1', 'a26_s2_t3', 'a26_s1_t2', 'a26_s6_t4', 'a26_s3_t2', 'a27_s2_t1', 'a27_s1_t2', 'a27_s1_t4', 'a27_s3_t1']
    BASIC_PATH = '/code/MMFederated/FML/sample-code-UTD/UTD-data/split-6-2/train'
    # step1: make the folder
    os.makedirs(join(BASIC_PATH,train_folder),exist_ok=True)
    os.makedirs(join(BASIC_PATH,train_folder,'label'),exist_ok=True)
    os.makedirs(join(BASIC_PATH,train_folder,'unlabel'),exist_ok=True)
    os.makedirs(join(BASIC_PATH,train_folder,'label','inertial'),exist_ok=True)
    os.makedirs(join(BASIC_PATH,train_folder,'label','skeleton'),exist_ok=True)
    os.makedirs(join(BASIC_PATH,train_folder,'unlabel','inertial'),exist_ok=True)
    os.makedirs(join(BASIC_PATH,train_folder,'unlabel','skeleton'),exist_ok=True)

    inertial_files = os.listdir(join(BASIC_PATH,'total','inertial'))
    skeleton_files = os.listdir(join(BASIC_PATH,'total','skeleton'))
    assert len(inertial_files) == len(skeleton_files)

    for inertial_file in inertial_files:
        identification = inertial_file.replace('_inertial.npy','')
        file_name = join(BASIC_PATH,'total','inertial',inertial_file)
        if identification in label_file_name:
            shutil.copy(file_name,join(BASIC_PATH,train_folder,'label','inertial'))
        else:
            shutil.copy(file_name,join(BASIC_PATH,train_folder,'unlabel','inertial'))
    for skeleton_file in skeleton_files:
        identification = skeleton_file.replace('_skeleton.npy','')
        file_name = join(BASIC_PATH,'total','skeleton',skeleton_file)
        if identification in label_file_name:
            shutil.copy(file_name,join(BASIC_PATH,train_folder,'label','skeleton'))
        else:
            shutil.copy(file_name,join(BASIC_PATH,train_folder,'unlabel','skeleton'))

    """random sampling"""
    # labeled_num_per_action = round(len(inertial_files) * label_rate / 2700)
    # for i in range(1,28):
    #     inertial_subclass_list = [item for item in inertial_files if item.split('_')[0] == f'a{i}']
    #     skeleton_subclass_list = [item for item in skeleton_files if item.split('_')[0] == f'a{i}']
    #     assert len(inertial_subclass_list) == len(skeleton_subclass_list)
    #     labeled_index = random.sample(range(len(inertial_subclass_list)),labeled_num_per_action)
    #     unlabeled_index = list(set(range(len(inertial_subclass_list)))-set(labeled_index))
    #     labeled_inertial_subclass_list = [inertial_subclass_list[i] for i in labeled_index]
    #     labeled_skeleton_subclass_list = [skeleton_subclass_list[i] for i in labeled_index]
    #     unlabeled_inertial_subclass_list = [inertial_subclass_list[i] for i in unlabeled_index]
    #     unlabeled_skeleton_subclass_list = [skeleton_subclass_list[i] for i in unlabeled_index]
    #     for item in labeled_inertial_subclass_list:
    #         file_name = join(BASIC_PATH,'total','inertial',item)
    #         shutil.copy(file_name,join(BASIC_PATH,f'label-27-{label_rate}-percent','label','inertial'))
    #     for item in labeled_skeleton_subclass_list:
    #         file_name = join(BASIC_PATH,'total','skeleton',item)
    #         shutil.copy(file_name,join(BASIC_PATH,f'label-27-{label_rate}-percent','label','skeleton'))

    #     for item in unlabeled_inertial_subclass_list:
    #         file_name = join(BASIC_PATH,'total','inertial',item)
    #         shutil.copy(file_name,join(BASIC_PATH,f'label-27-{label_rate}-percent','unlabel','inertial'))
    #     for item in unlabeled_skeleton_subclass_list:
    #         file_name = join(BASIC_PATH,'total','skeleton',item)
    #         shutil.copy(file_name,join(BASIC_PATH,f'label-27-{label_rate}-percent','unlabel','skeleton'))


if __name__ == '__main__':
    # inertial_files = os.listdir(join(TEST_DATA_PATH,'inertial'))
    # skeleton_files = os.listdir(join(TEST_DATA_PATH,'skeleton'))
    # # divide_by_user(inertial_files,skeleton_files)
    # # divide_by_class(inertial_files,skeleton_files)
    # print('len(inertial_files):', len(inertial_files))
    # print('len(skeleton_files):', len(skeleton_files))

    divide_label_and_unlabel_data(10)