# echo 'hello world'
# Back to the source folder: home/s2051012
cd /
cd /home/s2051012
. msc/toolchain.rc
. msc/venv/bin/activate
sshfs -o IdentityFile=/home/s2051012/msc/id_rsa -p 522 msc@129.215.91.172:/ /home/s2051012/msc/remote
cd msc/fyp-code/

python3 train_vq-model.py > vq-test.out


cd /
cd /home/s2051012
fusermount -u msc/remote
