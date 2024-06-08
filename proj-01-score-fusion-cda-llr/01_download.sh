#!/bin/bash
# 
# Download CSV file lists, pre-trained embeddings and so on.
srclink=
filename=SpeechSPC-mini-01-DATA.tar.gz
wget -q --show-progress ${srclink}

if [ -e ${filename} ];
then
    tar -xzvf ${filename}
else
    echo -e "Cannot download ${filename}"
fi

