#!/bin/bash
# Author: Z
# Version: 1.0
# Created Time: 2021/07/03
# Config the running environment of the APC model and train the APC in slurm

sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 10
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 20
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 30
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 40
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 50
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 60
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 70
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 80
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 90
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 100
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 200
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 300
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 400
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 500
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 600
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 700
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 800
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 900
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_raw ./test_kmeans_copy1.sh 1000
