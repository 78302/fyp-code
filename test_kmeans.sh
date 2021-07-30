#!/bin/bash
# Author: Z
# Version: 1.0
# Created Time: 2021/07/03
# Config the running environment of the APC model and train the APC in slurm


# echo 'hello world'
# Back to the source folder: home/s2051012
cd /
cd /home/s2051012
. msc/toolchain.rc
. msc/venv/bin/activate
sshfs -o IdentityFile=/home/s2051012/msc/id_rsa -p 522 msc@129.215.91.172:/ /home/s2051012/msc/remote
cd msc/fyp-code/
python3 testKM.py -n 'test_pre' -e 1 -t 0 -p './pretrain_model/model/Epoch50.pth.tar' -k $1 > job_k$1.out
cd /
cd /home/s2051012
fusermount -u msc/remote
# echo 'Successfully unmounted!'
# ls msc/remote/
