# Load data file paras
import argparse
# Deal with the use input parameters
parser = argparse.ArgumentParser(description='Parse the net paras')
parser.add_argument('--name', '-n', help='Name of the Model, required', required=True)
# required=True
args = parser.parse_args()

# Assign parameters
NAME = args.name

# import data from csv_file
import csv
# Load ceters and loss
import csv
cluster_bonds = []
loss = []
with open(NAME + '_result.csv', 'r') as csv_file:
    csvreader = csv.reader(csv_file)
    k = 0
    error = 0
    for l in csvreader:
        if int(l[0]) != k:
            cluster_bonds.append(k)
            loss.append(error)
        k = int(l[0])
        error = float(l[2])
    cluster_bonds.append(k)
    loss.append(error)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
x = cluster_bonds[1:]
y = loss[1:]

fig, ax = plt.subplots(figsize=(14,7))
ax.plot(x,y,'r--',label='Loss')

ax.set_title('Loss',fontsize=18)
ax.set_xlabel('Number of Clusters', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic')
ax.set_ylabel('loss', fontsize='x-large',fontstyle='oblique')
ax.legend()

plt.savefig("Clustering Loss.pdf")

# print(cluster_bonds, loss)





### DRAW classifier loss curve






### DRAW apc loss curve
