#!/bin/bash
# Author: Z
# Version: 1.0
# Created Time: 2021/07/03
# Config the running environment of the APC model and train the APC in slurm



cd /
cd /home/s2051012
. msc/toolchain.rc
. msc/venv/bin/activate
sshfs -o IdentityFile=/home/s2051012/msc/id_rsa -p 522 msc@129.215.91.172:/ /home/s2051012/msc/remote

cd msc/fyp-code/
# python3 run_apc.py
# python3 run_vqapc.py -n 'WSJ_VQAPC_50epochs' -lr 0.001 -e 50 -t 0 > job_train_apc.out
python3 run_vqapc.py -n 'VQ-test' -lr 0.001 -e 1 -t 0 -o 1 > job_train_vqapc_test.out



cd /
cd /home/s2051012
fusermount -u msc/remote
