#!/bin/bash
# Author: Z
# Version: 1.0
# Created Time: 2021/07/03
# Config the running environment of the APC model and train the APC in slurm


# echo 'hello world'
# Back to the source folder: home/s2051012
cd /
cd /home/s2051012
# Activate the env?
. msc/toolchain.rc
. msc/venv/bin/activate

# ls

# Mount the shared file
sshfs -o IdentityFile=/home/s2051012/msc/id_rsa -p 522 msc@129.215.91.172:/ /home/s2051012/msc/remote
# echo 'Successfully mounted the share file under remote folder!'
# ls msc/remote/

# Run the code
# ls msc/fyp-code/
cd msc/fyp-code/
# python3 run_apc.py
python3 run_classifier.py -n 'UsePretrain-lr0001' -lr 0.001 -is 512 -t 0 -o $1 > job_classifier_raw_lr0001_epoch$1.out

# Unmount it
cd /
cd /home/s2051012
fusermount -u msc/remote
# echo 'Successfully unmounted!'
# ls msc/remote/
