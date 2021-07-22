# Load data file paras
import argparse
# Deal with the use input parameters
parser = argparse.ArgumentParser(description='Parse the net paras')
parser.add_argument('--name', '-n', help='Name of the Model, required', required=True)
parser.add_argument('--target', '-t', help='Name of the Model, required', type=int, required=True)
# required=True
args = parser.parse_args()

# Assign parameters
NAME = args.name
TARGET = args.target

if TARGET == 1:
    #### Kmeans plot
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




if TARGET == 2:
    ### DRAW classifier loss curve
    # Draw the train loss
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # load data
    e_results = np.load(NAME + '_results.npy')

    y1 = e_results[0]
    y2 = e_results[1]
    x = np.arange(1,21)
    fig, ax = plt.subplots(figsize=(14,7))
    ax.plot(x,y1,'r',label='Train Loss')
    ax.plot(x,y2,'b',label='Dev Loss')

    ax.set_title('Loss',fontsize=18)
    ax.set_xlabel('Epoch', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic')
    ax.set_ylabel('Loss', fontsize='x-large',fontstyle='oblique')
    ax.legend()

    plt.savefig("{:s}_classifier_loss.pdf".format(NAME))

    y1 = e_results[2]
    y2 = e_results[3]
    x = np.arange(1,21)
    fig2, ax2 = plt.subplots(figsize=(14,7))
    ax2.plot(x,y1,'r--',label='Train Err')
    ax2.plot(x,y2,'r--',label='Dev Err')

    ax2.set_title('Error Rate',fontsize=18)
    ax2.set_xlabel('Epoch', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic')
    ax2.set_ylabel('Error Rate', fontsize='x-large',fontstyle='oblique')
    ax2.legend()

    plt.savefig("{:s}_classifier_err.pdf".format(NAME))



if TARGET == 3:
    ### DRAW apc loss curve
    # Draw the train loss
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Load _losses
    losses = np.load(NAME + '_losses.npy')

    y_t = losses[0]
    y_d = losses[1]
    x = np.arange(1,101)
    fig, ax = plt.subplots(figsize=(14,7))
    ax.plot(x,y_t,'r',label='Train Loss')
    ax.plot(x,y_d,'b',label='Dev Loss')

    ax.set_title('Loss',fontsize=18)
    ax.set_xlabel('epoch', fontsize=18,fontfamily = 'sans-serif',fontstyle='italic')
    ax.set_ylabel('loss', fontsize='x-large',fontstyle='oblique')
    ax.legend()

    plt.savefig("{:s}_apc_loss.pdf".format(NAME))
