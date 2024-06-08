#!/usr/bin/python3
"""
"""
from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import random
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import speechbrain as sb

from tqdm.contrib import tqdm

from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main

import datasets
import model

__author__ = "Xin Wang"
__email__ = "wangxin@nii.ac.jp"
__copyright__ = "Copyright 2023, Xin Wang"


def analysis(hparams_file, run_opts, overrides):

    # Load hyperparameters file with command-line overrides   
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    dev_dataset = datasets.create_dev_dataset(hparams)
    eval_dataset = datasets.create_eval_dataset(hparams)
    
    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    sasvbaseline = model.SASVBaseline(
        modules= model.SASVBaseline.create_modules(hparams['model_config']),
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    
    # load the best checkpoint
    sasvbaseline.load_checkpoint()

    # Analysis
    # score of dev set
    sasvbaseline.analysis(
        dev_dataset,
        hparams['dataloader_dev_options'],
        hparams['analysis_dump_folder']
    )

    # score of test set
    sasvbaseline.analysis(
        eval_dataset,
        hparams['dataloader_eval_options'],
        hparams['analysis_dump_eval_folder']
    )
    return
    
    

def train(hparams_file, run_opts, overrides):

    # Initialize ddp (useful only for multi-GPU DDP training) 
    sb.utils.distributed.ddp_init_group(run_opts)
    
    # Load hyperparameters file with command-line overrides   
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Dataset IO prep: creating Dataset objects and proper encodings for phones
    train_dataset = datasets.create_trn_dataset(hparams)
    dev_dataset = datasets.create_dev_dataset(hparams)
    eval_dataset = datasets.create_eval_dataset(hparams)

    # 
    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    sasvbaseline = model.SASVBaseline(
        modules= model.SASVBaseline.create_modules(hparams['model_config']),
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    sasvbaseline.fit(
        sasvbaseline.hparams.epoch_counter,
        train_dataset,
        dev_dataset,
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_dev_options"],
    )

    # load the best checkpoint
    sasvbaseline.load_checkpoint()

    # Testing
    sasvbaseline.test(
        eval_dataset,
        hparams['dataloader_dev_options']
    )
    return

if __name__ == "__main__":

    # 
    torch.backends.cudnn.benchmark = True

    if sys.argv[1] == 'analysis':
        hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[2:])
        analysis(hparams_file, run_opts, overrides)

    else:
        hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
        train(hparams_file, run_opts, overrides)
