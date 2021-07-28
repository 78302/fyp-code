#!/bin/bash
# Author: Z
# Version: 1.0
# Created Time: 2021/07/03
# Config the running environment of the APC model and train the APC in slurm
sbatch -p Teach-Standard -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 10
sbatch -p Teach-Standard -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 20
sbatch -p Teach-Standard -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 30
sbatch -p Teach-Standard -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 40
sbatch -p Teach-Standard -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 50
sbatch -p Teach-Standard -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 60
sbatch -p Teach-Standard -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 70
sbatch -p Teach-Standard -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 80
sbatch -p Teach-Standard -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 90
sbatch -p Teach-Standard -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 100
# sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 200
# sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 300
# sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 400
# sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 500
# sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 600
# sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 700
# sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 800
# sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 900
# sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 1000
# sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 2000
# sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 3000
# sbatch -p General_Usage -c 4 -d singleton -J job_kmeans_pre ./test_kmeans_copy2.sh 4000
