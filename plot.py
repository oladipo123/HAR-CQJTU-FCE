#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : plot.py
# @Description :
import argparse

import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot

SENSORS = ["Accelerometer", "Gyroscope", "Magnetometer"]
SENSOR_NAMES = ['ACC-X', 'ACC-Y', 'ACC-Z', 'GYRO-X', 'GYRO-Y', 'GYRO-Z', 'MAG-X', 'MAG-Y', 'MAG-Z']
SENSOR_NAMES1 = ['MASK ACC-X', 'MASK ACC-Y', 'MASK ACC-Z', 'MASK GYRO-X', 'MASK GYRO-Y', 'MASK GYRO-Z', 'MASK MAG-X', 'MASK MAG-Y', 'MASK MAG-Z']
COLOR_BLUE = 'tab:blue'; COLOR_ORANGE = 'tab:orange'; COLOR_GREEN = 'tab:green'
COLOR_RED = 'tab:red'; COLOR_PURPLE = 'tab:purple'
COLOR_LIST = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_RED]
LINE_STYLES = ['solid', 'dotted']


def plot_tsne(data, labels, dimension=2, label_names=None):
    tsne = TSNE(n_components=dimension)
    data_ = tsne.fit_transform(data)
    # print(data_)
    ls = np.unique(labels)
    print(ls)
    plt.figure()
    bwith = 2
    TK = plt.gca()
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)
    for i in range(ls.size):
        index = labels == ls[i]
        x = data_[index, 0]
        y = data_[index, 1]
        if label_names is None:
            plt.scatter(x, y, label=str(int(ls[i])))
        else:
            plt.scatter(x, y, label=label_names[int(ls[i])])
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='upper left', prop={'size': 8.5}) #, prop={'size': 20, 'weight':'bold'}
    plt.show()
    return data_


def plot_pca(data, labels, dimension=2):
    pca = PCA(n_components=dimension)
    data_ = pca.fit_transform(data)
    ls = np.unique(labels)
    plt.figure()
    for i in range(ls.size):
        index = labels == ls[i]
        x = data_[index, 0]
        y = data_[index, 1]
        plt.scatter(x, y, label=str(ls[i]))
    plt.show()
    # plt.close()


def plot_matrix(matrix, labels_name=None):
    plt.figure()
    row_sum = matrix.sum(axis=1)
    matrix_per = np.copy(matrix).astype('float')
    for i in range(row_sum.size):
        if row_sum[i] != 0:
            matrix_per[i] = matrix_per[i] / row_sum[i]
    # plt.figure(figsize=(10, 7))
    if labels_name is None:
        labels_name = "auto"
    sn.heatmap(matrix_per, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels_name, yticklabels=labels_name)
    plt.xticks(rotation=0)  # 水平显示 x 轴标签
    plt.yticks(rotation=0)  # 水平显示 y 轴标签（可选）
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    # plt.savefig()
    return matrix


def plot_embedding(embeddings, labels, label_index=0, reduce=1000, label_names=None):
    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2])
    index_rand = np.arange(embeddings.shape[0])
    np.random.shuffle(index_rand)
    index_rand = index_rand[:reduce]
    if isinstance(label_index, list):
        label_composite = np.zeros(labels.shape[0])
        for i in range(len(label_index)):
            label_composite += labels[:, 0, label_index[i]] * pow(10, len(label_index) - 1 - i)
        plot_tsne(embeddings[index_rand, :], label_composite[index_rand])
        return None
    else:
        data_tsne = plot_tsne(embeddings[index_rand, :], labels[index_rand, 0, label_index], label_names=label_names)
        return data_tsne, labels[index_rand, 0, label_index]
        # plot_pca(embeddings[index_rand, :], labels[index_rand, label_index])


def plot_reconstruct_sensor(sensors, sensors_re, sensor_dimen=3):
    sensor_num = sensors.shape[1] // 3
    # print(sensor_num)

    # fig, axs = plt.subplots(sensor_num*3)
    fig, axs = plt.subplots(3, 2, figsize=(15, 5 * 3))
    fig.suptitle('IMU Sensor Data') #Sensor Reconstruction Comparison
    x = np.arange(sensors.shape[0])
    # plt.ion()
    for i in range(sensor_num):
        index_start = i * sensor_dimen
        for j in range(sensor_dimen):
            axs[j, i].set_xlabel("Index")
            axs[j, i].set_ylabel(SENSORS[i])
            dimen = index_start + j
            axs[j, i].plot(x, sensors[:, dimen], label=SENSOR_NAMES[dimen], linestyle=LINE_STYLES[0], color=COLOR_LIST[j]) #
            axs[j, i].plot(x, sensors_re[:, dimen], label=SENSOR_NAMES1[dimen], linestyle=LINE_STYLES[1], linewidth=3, color=COLOR_LIST[j])
            axs[j, i].legend()
    plt.show()


def plot_roc_auc(y_pred, y_true):
    auc = metrics.roc_auc_score(y_true, y_pred)
    print('ROC AUC=%.3f' % (auc))
    fpr, tpr, thre = metrics.roc_curve(y_true, y_pred)
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.show()
    return fpr, tpr, thre

def plot_predict(ori_seq, seqs, seq_recon, masked_pos):
    x = np.arange(ori_seq.shape[1])
    masked_pos = masked_pos.detach().numpy()
    # print(masked_pos.shape)
    seq_recon = seq_recon.detach().numpy()
    # print(seq_recon.shape)
    fig, ax = plt.subplots()
    # for batch_index in range(ori_seq.shape[0]):
    y = ori_seq[0]
    ax.plot(x, y[:, 1], label='ACC-X')
    ax.scatter(masked_pos[0], seqs[0, :, 1], s=50, c='red', marker='o', label='Maked ACC-X')  # s是点的大小，c是颜色，marker='o'表示圆形
    ax.scatter(masked_pos[0], seq_recon[0, :, 1], s=50, c='blue', marker='o', label='Predicted ACC-X')
    ax.legend()
    plt.show()



# if __name__ == "__main__":
