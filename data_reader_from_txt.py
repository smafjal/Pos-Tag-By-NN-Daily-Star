#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os,codecs
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

network_input_test="network_input/test/"
network_input_train="network_input/train/"
taglist=['N','V','J','D','L','A','C','RD','P','PU','PP']

def read_data(path):
    with open(path,"r") as r:
        vector=[]
        cnt=0;
        for x in r.readlines():
            lis=[float(i) for i in x.split(" ")]
            print len(lis)
            if cnt>10:
                break
            cnt=cnt+1



def main():
    rpath1="network_input/train/tag_vector_01.txt"
    rpath2="network_input/train/word_vector_03.txt"
    read_data(rpath2)

if __name__=="__main__":
    main()
