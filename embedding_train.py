#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sets import Set
import numpy as np
import os,codecs
import pickle
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

corpus_path="daily_star_corpus" # all files path [ folder name]
# corpus_path ="/media/1TB/legion/piData/thedailystar"
#corpus_path ="/media/1TB/legion/piData/testData"
model_path="daily_star_save_model" # where i save model
data_path="input_tt" # data for testing
network_in="network_input"
# taglist=['N','V','J','D','L','A','C','RD','P','PU','PP']
taglist=['PRP$', 'VBG', 'VBD', 'VBN', 'VBP', 'WDT', 'JJ', 'WP', 'VBZ', 'DT', 'RP', 'NN', 'FW', 'POS', 'TO', 'PRP', 'RB', 'NNS', 'NNP', 'VB', 'WRB', 'CC', 'PDT', 'RBS', 'RBR', 'CD', 'EX', 'IN', 'WP$', 'MD', 'NNPS', 'JJS', 'JJR', 'SYM', 'UH']

tagset=Set()

def chomps(s):
    return s.rstrip('\n')

def get_unicode(input):
    input=chomps(input)
    if type(input) != unicode:
        input =  input.decode('utf-8')
        return input
    else:
         return input
def get_ascii(input):
    if type(input) != unicode:
        return input
    else:
        return input.encode("utf-8")

# iterator that is used for all file reader
# it returns word list vector at each sentence
class MySentences(object):
    def __init__(self,dirname):
        self.dirname=dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname,fname)):
                lines=[x for x in line.split()]
                lines=[x for x in lines]
                lines.insert(0,"<PAD>")
                lines.insert(-1,"</PAD>")
                yield lines

def do_embedding(path,model_path):
    print '----[start embedding on data corpus ]---'
    # word2vec parameters
    min_count=1 # word frequency grater or equal 'min_count' can be embedded
    size=100 # word vector size.simply known as 'embedding size'
    workers=16
    window = 4 # contexual window

    # a memory-friendly iterator
    sentences = MySentences(path)
    model = gensim.models.Word2Vec(sentences,
        min_count=min_count,
        size=size,
        workers=workers,
        window=window
    )
    save_p=model_path+'/model_corpus'
    model.save(save_p)
    return model,save_p

def retrain_model(model_path,corpus_path):
    sentences=MySentences(corpus_path)
    new_model = gensim.models.Word2Vec.load(model_path)
    new_model.train(sentences)
    return new_model

def tag_hot_vec(val):
    # print "Tag----------> ",val
    tag_len=len(taglist)
    y=[0]*tag_len
    for i in range(len(taglist)):
        if get_unicode(taglist[i]) == get_unicode(val):
            y[i]=1
    return y

class MySentencesTest(object):
    def __init__(self,dirname):
        self.dirname=dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname,fname)):
                # print "Line: ",line
                lines=[x for x in line.split()]
                lines=[x for x in lines]
                yield lines

def make_sequence(model,sentence,X,Y,line,window=1):
    end_pad=model["</PAD>"]
    window_mid=window/2

    only_sen=[]
    only_tag=[]


    for i in range(len(sentence)):
        if i%2 == 0:
            only_sen.append(sentence[i])
        else:
            only_tag.append(sentence[i])
            tagset.add(sentence[i])
            # if sentence[i] =="#" or sentence[i] =="." or sentence[i] ==":" or sentence[i] ==",":
                # print "Vul Tag: ",sentence[i], line,i

    sen_len=len(only_sen)
    for i in range(sen_len-window_mid):
        seq_x=[]
        tag=""
        for j in range(window):
            if i+j<sen_len:
                word=only_sen[i+j]
                seq_x.append(model[word])
            else:
                seq_x.append(end_pad)

            if j == window_mid:
                tag = only_tag[i+j]
            pass
        seq_x=np.reshape(seq_x,-1)
        seq_y=tag_hot_vec(tag)

        X.append(seq_x)
        Y.append(seq_y)
        pass

def generate_em_vector(model,path,window=1):
    print "<>"*34

    print "Path: ",path
    sentences=MySentencesTest(path)
    X=[]
    Y=[]
    cnt=1
    for x in sentences:
       make_sequence(model,x,X,Y,cnt,window)
       cnt=cnt+1

    X_arr=np.asarray(X).astype(np.float32)
    Y_arr=np.asarray(Y).astype(np.float32)

    return X_arr,Y_arr


def write_2D_data(data,path):
    with open(path,"w") as w:
        w.writelines(" ".join( str(j) for j in i) + "\n" for i in data)
        # w.writelines(','.join(str(j) for j in i) + '\n' for i in data)

def save_em_vector(X,Y,path):
    p1=path+"/testX.npy"
    p2=path+"/testY.npy"
    np.save(p1,X)
    np.save(p2,Y)

def load_npy(path):
    p1=path+"/testX.npy"
    p2=path+"/testY.npy"
    X=np.load(p1)
    Y=np.load(p2)
    return X,Y



def main():
    model,saved_model_path=do_embedding(corpus_path,model_path)
    print "Model Saved On: ",saved_model_path

    # new_corpus=""
    # model=retrain_model(model,new_corpus)

    X,Y=generate_em_vector(model,data_path,1)
    save_em_vector(X,Y,network_in)
    print "len-X: ",len(X)
    print "len-Y: ",len(Y)
    print "Tagset - len: ",len(tagset)
    print tagset

    X,Y=load_npy(network_in)
    print "X-shape:-->",X.shape
    print "Y-shape:-->",Y.shape
    print "*"*80
    print X

if __name__=="__main__":
    main()

