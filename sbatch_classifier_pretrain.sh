#!/bin/bash
# Author: Z
# Version: 1.0
# Created Time: 2021/07/03
# Config the running environment of the APC model and train the APC in slurm
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 1
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 2
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 3
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 4
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 5
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 6
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 7
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 8
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 9
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 10
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 11
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 12
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 13
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 14
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 15
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 16
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 17
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 18
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 19
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_pre ./job_raw_classifier_usePretrain2.sh 20
