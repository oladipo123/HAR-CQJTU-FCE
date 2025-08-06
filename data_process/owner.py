# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 28/4/2021
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : shoaib.py
# @Description : https://www.mdpi.com/1424-8220/14/6/10146

import os
import numpy as np
import pandas as pd


DATASET_PATH = r'D:\Owner1\Owner\DateSet'
ACT_LABELS = ["walking", "sitting", "standing", "jogging", "lying", "upstairs" , "downstairs"]
SAMPLE_WINDOW = 20


def label_name_to_index(label_names):
    label_index = np.zeros(label_names.size)
    for i in range(len(ACT_LABELS)):
        ind = label_names == ACT_LABELS[i]
        # print(np.sum(ind))
        label_index[ind] = i
    return label_index


def down_sample(data, window_target):
    window_sample = window_target * 1.0 / SAMPLE_WINDOW
    result = []
    if window_sample.is_integer():
        window = int(window_sample)
        for i in range(0, len(data), window):
            slice = data[i: i + window, :]
            result.append(np.mean(slice, 0))
    else:
        window = int(window_sample)
        remainder = 0.0
        i = 0
        while 0 <= i + window + 1 <= data.shape[0]:
            remainder += window_sample - window
            if remainder >= 1:
                remainder -= 1
                slice = data[i: i + window + 1, :]
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window +  1, i,  i+window))
                result.append(np.mean(slice, 0))
                i += window + 1
            else:
                slice = data[i: i + window, :]
                result.append(np.mean(slice, 0))
                # print('i: %d, window: %d, start: %d, end: %d' % (i, window +  1, i,  i+window))
                i += window
    return np.array(result)


def preprocess(path, path_save, version, target_window=50, seq_len=20, position_num=5):
    data = []
    label = []
    for root, dirs, files in os.walk(path):
        # print(f"the root is {root}")
        # print(f"the dirs is {dirs}")
        # print(f"the files  is {files}")
        for f in range(len(files)):
            if 'Participant' in files[f]:
                exp = pd.read_csv(os.path.join(root, files[f]), skiprows=1)

                labels_activity = exp.iloc[:, -1].to_numpy()
                # print(labels_activity)
                labels_activity = label_name_to_index(labels_activity)
                # print(labels_activity)
                for a in range(len(ACT_LABELS)):
                    exp_act = exp.iloc[labels_activity == a, :]
                    for i in range(position_num):
                        index = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) + i * 13
                        exp_pos = exp_act.iloc[:, index].to_numpy(dtype=np.float32)
                        print("User-%s, activity-%s, position-%d: num-%d" % (files[f], ACT_LABELS[a], i, exp_pos.shape[0]))
                        if exp_pos.shape[0] > 0:
                            exp_pos_down = down_sample(exp_pos, target_window)
                            print(exp_pos_down.shape)
                            sensor_down = exp_pos_down[:exp_pos_down.shape[0] // seq_len * seq_len, :]
                            print(sensor_down.shape)
                            sensor_down = sensor_down.reshape(sensor_down.shape[0] // seq_len, seq_len, sensor_down.shape[1])
                            print(sensor_down.shape)
                            sensor_label = np.multiply(np.ones((sensor_down.shape[0], sensor_down.shape[1], 1)),
                                                       np.array([a, i, f]).reshape(1, 3))

                            data.append(sensor_down)
                            label.append(sensor_label)

    data = np.concatenate(data, 0)
    label = np.concatenate(label, 0)
    print('All label processed. Size: %d' % (label.shape[0]))
    print('All data processed. Size: %d' % (data.shape[0]))
    np.save(os.path.join(path_save, 'data_' + version + '.npy'), np.array(data))
    np.save(os.path.join(path_save, 'label_' + version + '.npy'), np.array(label))
    return data, label


path_save = r'owner'
version = r'20_120'
data, label = preprocess(DATASET_PATH, path_save, version, seq_len=120)