#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : pretrain.py
# @Description :
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, TensorDataset, DataLoader
from tensorboardX import SummaryWriter

import models, train
from config import MaskConfig, TrainConfig, PretrainModelConfig
from models import LIMUBertModel4Pretrain
from plot import plot_reconstruct_sensor, plot_embedding, plot_predict
from utils import set_seeds, get_device \
    , LIBERTDataset4Pretrain, handle_argv, load_pretrain_data_config, prepare_classifier_dataset, \
    prepare_pretrain_dataset, Preprocess4Normalization,  Preprocess4Mask


def main(args, training_rate):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)
    # print(data.shape)

    pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Mask(mask_cfg)]
    pipeline1 = [Preprocess4Normalization(model_cfg.feature_num)]
    # pipeline = [Preprocess4Mask(mask_cfg)]
    data_train, label_train, data_test, label_test = prepare_pretrain_dataset(data, labels, training_rate, seed=train_cfg.seed)
    # print(data_train.shape)
    data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline, pipeline1=pipeline1)
    data_set_test = LIBERTDataset4Pretrain(data_test, pipeline=pipeline,pipeline1=pipeline1)
    # for batch in data_set_train:
    #     mask_seq, masked_pos, seq = batch
    #
    #     # 打印变量类型
    #     print(f"the mask_seq shape is {mask_seq.shape}")  # 打印 mask_seq 的类型
    #     print(f"the masked_pos shape is {masked_pos.shape}")  # 打印 masked_pos 的类型
    #     print(f"the seq shape is {seq.shape}")  # 打印 seq 的类型
    num_batches = 0
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    # for batch in data_loader_train:
    #     mask_seq, masked_pos, seq, ori_seq = batch
    #
    #     # 打印变量类型
    #     print(f"the mask_seq shape is {mask_seq.shape}")  # 打印 mask_seq 的类型
    #     print(f"the masked_pos shape is {masked_pos.shape}")  # 打印 masked_pos 的类型
    #     print(f"the seq shape is {seq.shape}")  # 打印 seq 的类型
    #     print(f"the ori_seq shape is {ori_seq.shape}")  # 打印 ori_seq 的类型
    #     num_batches += 1
    # print(f'The DataLoader iterates {num_batches} times per epoch.')
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    model = LIMUBertModel4Pretrain(model_cfg)

    criterion = nn.MSELoss(reduction='none')

    # use_warmup_optimizer=True
    # scale_factor = 1
    # warmup_steps = 20
    # if use_warmup_optimizer:
    #     optimizer = WarmupOptimizer(Adam(params.model.parameters(), lr=1e-3), d_model, scale_factor, warmup_steps)
    # else:
    #     optimizer = Adam(params.model.parameters(), lr=1e-3)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer = train.Trainer(train_cfg, model, optimizer, args.save_path, device)

    def func_loss(model, batch, label):
        mask_seqs, masked_pos, seqs, ori_seq = batch
        # print(f"seq is {seqs.shape}")

        seq_recon = model(mask_seqs, masked_pos)
        # if label == 1:
        #     plot_predict(ori_seq, seqs, seq_recon, masked_pos)
        # plot_predict(ori_seq, seqs, seq_recon, masked_pos)
        # print(f"the seq_recon shape is {seq_recon.shape}")
        loss_lm = criterion(seq_recon, seqs) # for masked LM
        return loss_lm

    def func_forward(model, batch):
        mask_seqs, masked_pos, seqs, ori_seq = batch
        seq_recon = model(mask_seqs, masked_pos)
        return seq_recon, seqs

    def func_evaluate(seqs, predict_seqs):
        loss_lm = criterion(predict_seqs, seqs)
        return loss_lm.mean().cpu().numpy()

    if hasattr(args, 'pretrain_model'):
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test
                      , model_file=args.pretrain_model)
    else:
        trainer.pretrain(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_test, model_file=None)


if __name__ == "__main__":
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    training_rate = 0.8
    main(args, training_rate)
