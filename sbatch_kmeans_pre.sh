#!/bin/bash
# Author: Z
# Version: 1.0
# Created Time: 2021/07/03
# Config the running environment of the APC model and train the APC in slurm


sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 200
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 300
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 400
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 500
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 600
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 700
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 800
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 900
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 1000
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 1100
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 2000
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 3000
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 4000
sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 5000
