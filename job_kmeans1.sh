#!/bin/bash
# Author: Z
# Version: 1.0
# Created Time: 2021/07/03
# Config the running environment of the APC model and train the APC in slurm



# Back to the source folder: home/s2051012
cd /
cd /home/s2051012
. msc/toolchain.rc
. msc/venv/bin/activate
# Mount the shared file
sshfs -o IdentityFile=/home/s2051012/msc/id_rsa -p 522 msc@129.215.91.172:/ /home/s2051012/msc/remote

# Run the code
cd msc/fyp-code/
python3 kMeans.py -n 'raw' -e $1 -t 0 -k $2  > job_kmeans_raw_e$1_k$2.out
# python3 kMeans.py -n 'pretrain' -e 1 -k 5 -p './pretrain_model/model/Epoch50.pth.tar' -s './data/raw_fbank_train_si284.scp' > job_kmeans_pretrain_e1_k5.out

cd /
cd /home/s2051012
fusermount -u msc/remote
