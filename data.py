



import numpy as np
import pickle as pkl
import scipy.sparse as sp
import networkx as nx
import sys
import json


def load_dataset(dataset_directory):
    dataset = dataset_directory
    names = ['feature', 'label', 'graph', 'idx_train', 'idx_eval', 'idx_test']
    objects = []
    for i in range(len(names)):
        f = open("./graphHAT/data/{}/{}.bin".format(dataset, names[i]), 'rb')
        if sys.version_info > (3, 0): # if python==3.x
            objects.append(pkl.load(f, encoding='latin1'))
        else: # if python==2.x
            objects.append(pkl.load(f))


    return objects

def write_file(directory):
    features, labels, graph, idx_train, idx_val, idx_test = load_dataset(directory)

    features = features.toarray()
    with open(directory + "/" + directory + '.features', 'w') as file:
        for i, line in enumerate(list(features)):
            line_str = str(i) + " " + " ".join([str(line) for line in list(line)])
            file.write(line_str + "\n")
    file.close()

    g = nx.Graph(graph)
    nx.write_edgelist(g, directory + "/" + directory + "_edgelist.txt", data=False)
    with open(directory + "/" + directory + '_labels.txt', 'w') as file:
        for i, array in enumerate(list(labels)):
            file.write(str(i) + " " + str(np.argmax(array)) + "\n")
    file.close()
    

# features, labels, graph, idx_train, idx_val, idx_test = load_dataset("cora")
# print(idx_test)