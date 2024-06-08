#!/bin/bash

system=$1
datafolder=$PWD/DATA

if [ "$#" -ne 1 ]; then
    echo -e "please execute: bash 02_train.py SYSTEM_NAME"
    exit
fi

cd ${system}
python main.py configs/sys_emb.yaml --data_folder ${datafolder}
echo "Training finished."
