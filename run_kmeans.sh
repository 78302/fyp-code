cd /
cd /home/s2051012
. msc/toolchain.rc
. msc/venv/bin/activate
sshfs -o IdentityFile=/home/s2051012/msc/id_rsa -p 522 msc@129.215.91.172:/ /home/s2051012/msc/remote
cd msc/fyp-code/
python3 kmeans.py -n 'km_rawLSTM' -e 10 -t 0 -p './pretrain_model/model/Epoch0.pth.tar' -k $1 > job_km_rawLSTM_k$1.out
cd /
cd /home/s2051012
fusermount -u msc/remote
