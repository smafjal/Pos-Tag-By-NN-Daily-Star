#!/usr/bin/env python
import numpy as np
import os,codecs
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

network_input_path="network_input"

def load_data(path):
    data=np.load(path)
    return data

def read_data(path):
    p1=path+"/testX.npy"
    p2=path+"/testY.npy"
    X=load_data(p1)
    Y=load_data(p2)
    return X,Y

def main():
    X,Y=read_data(network_input_path)
    print "Shape-X: ",X.shape
    print "Shape-Y: ",Y.shape

if __name__=="__main__":
    main()
