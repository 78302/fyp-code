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
python3 run_classifier.py -n 'Test' -lr 0.01 -e 1 -tu '../remote/data/wsj/extra/si284-0.9-train.fbank.scp' -tl '../remote/data/wsj/extra/si284-0.9-train.bpali.scp' -tp '../remote/data/wsj/extra/train-si284.bpali' -du '../remote/data/wsj/extra/si284-0.9-dev.fbank.scp' -dl '../remote/data/wsj/extra/si284-0.9-dev.bpali.scp' -dp '../remote/data/wsj/extra/si284-0.9-dev.bpali' > job_raw_classifier.out
# echo 'Model pretrained!'

# Unmount it
cd /
cd /home/s2051012
fusermount -u msc/remote
# echo 'Successfully unmounted!'
# ls msc/remote/
