#!/usr/bin/bash

# Install the environment
ENVNAME=sb

eval "$(conda shell.bash hook)"
conda activate ${ENVNAME}
retVal=$?

if [ $retVal -ne 0 ]; then
    echo "Install conda environment ${ENVNAME}"

    # conda env
    conda create -n ${ENVNAME} python=3.8 pip --yes
    conda activate ${ENVNAME}
    
    git clone https://github.com/speechbrain/speechbrain.git
    cd speechbrain

    # i used this specific checkpoint on github
    git checkout 16ef03604b187ff1d926368963bfda09515a47f0
    # following the official guide
    pip install -r requirements.txt
    pip install --editable .

    # install other tools
    conda install -y -c conda-forge scikit-learn=1.3.2
    conda install -y -c conda-forge pandas
    echo "Conda environment ${ENVNAME} has been installed"
else
    echo "Conda environment ${ENVNAME} has been installed"
fi
