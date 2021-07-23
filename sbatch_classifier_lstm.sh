#!/bin/bash
# Author: Z
# Version: 1.0
# Created Time: 2021/07/03
# Config the running environment of the APC model and train the APC in slurm
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 1
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 2
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 3
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 4
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 5
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 6
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 7
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 8
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 9
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 10
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 11
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 12
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 13
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 14
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 15
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 16
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 17
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 18
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 19
sbatch -p Teach-Standard --gres=gpu:1 -d singleton -J job_classifier_lstm ./job_raw_classifier_lstm.sh 20
