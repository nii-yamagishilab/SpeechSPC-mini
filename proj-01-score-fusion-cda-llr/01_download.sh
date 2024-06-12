#!/bin/bash
# 
# Download CSV file lists, pre-trained embeddings and so on.
srclink=https://www.dropbox.com/scl/fo/p7q1g87v8z114l1qgw5fd/ALLRzikiWKPcMtcjcUKNmps/SpeechSPC-mini-01-DATA.tar.gz?rlkey=mwdjclt4m90s8635n9q70y8cj
filename=SpeechSPC-mini-01-DATA.tar.gz
wget -q -O ${filename} --show-progress ${srclink}

if [ -e ${filename} ];
then
    tar -xzvf ${filename}
else
    echo -e "Cannot download ${filename}"
fi
