#!/bin/bash
# Author: Z
# Version: 1.0
# Created Time: 2021/07/03
# Config the running environment of the APC model and train the APC in slurm

# if [ $1 -eq 1 ] ; then
#     echo "APC"
#     exit 1
#
# elif [ $1 -eq 2 ] ; then
#     echo "classifier"
#     exit 1
#
# elif [ $1 -eq 3 ] ; then
#     echo "Kmeans"
#     exit 1
#
# else
#     echo "1 for apc, 2 for probing, 3 for kmeans"
# fi

# apc-training
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 1
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 2
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 3
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 4
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 5
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 6
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 7
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 8
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 9
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 10
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 11
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 12
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 13
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 14
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 15
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 16
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 17
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 18
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 19
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 20
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 21
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 22
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 23
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 24
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 25
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 26
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 27
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 28
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 29
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 30
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 31
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 32
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 33
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 34
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 35
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 36
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 37
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 38
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 39
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 40
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 41
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 42
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 43
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 44
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 45
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 46
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 47
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 48
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 49
sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 50
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 51
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 52
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 53
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 54
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 55
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 56
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 57
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 58
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 59
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 60
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 61
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 62
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 63
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 64
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 65
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 66
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 67
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 68
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 69
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 70
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 71
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 72
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 73
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 74
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 75
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 76
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 77
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 78
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 79
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 80
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 81
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 82
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 83
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 84
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 85
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 86
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 87
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 88
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 89
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 90
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 91
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 92
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 93
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 94
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 95
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 96
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 97
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 98
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 99
# sbatch -p Teach-Short --gres=gpu:1 -d singleton -J job_train_apc ./job1.sh 100
