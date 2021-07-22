#!/bin/bash
# Author: Z
# Version: 1.0
# Created Time: 2021/07/03
# Config the running environment of the APC model and train the APC in slurm
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 1
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 2
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 3
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 4
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 5
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 6
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 7
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 8
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 9
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 10
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 11
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 12
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 13
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 14
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 15
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 16
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 17
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 18
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 19
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_raw ./job_raw_classifier.sh 20
