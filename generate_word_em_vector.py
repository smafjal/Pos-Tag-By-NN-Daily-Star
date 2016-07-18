#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os,codecs
import gensim, logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

bd_corpus_path="bd_corpus" # all files path [ folder name]
saved_model_path="bd_save_model/model_corpus_bd" # where i save model
data_path="input/taged_sentence.txt" # data for testing
network_input_test="network_input/test/"
network_input_train="network_input/train/"
taglist=['N','V','J','D','L','A','C','RD','P','PU','PP']


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

def convert_to_min_tag(tag):
    # if(len(tag)==1):
        # print tag
    fc=tag[0]
    sc=tag[1]
    if fc != "P":
        if fc=="R":
            return "RD"
        return fc
    elif sc == 'U':
        return "PU"
    elif len(tag)==2:
        return "PP"
    else:
        return "P"
    pass

def tag_hot_vec(val):
    # print "Tag----------> ",val
    tag_len=len(taglist)
    y=[0]*tag_len
    for i in range(len(taglist)):
        if get_unicode(taglist[i]) == get_unicode(val):
            y[i]=1
    return y

def load_model(model_path):
    model=gensim.models.Word2Vec.load(model_path)
    return model

def sen_parse(path):
    rfile=codecs.open(path, "r", "utf-8")
    rdata=rfile.readlines()

    sen_list_train_word=[]
    sen_list_train_tag=[]
    sen_list_test_word=[]
    sen_list_test_tag=[]

    # iterate line
    for i in range(len(rdata)):
        str_i=rdata[i]
        if len(str_i)==1: # for newline char problem
            continue
        str_i=str_i.strip().split(' ')
        x_words=[]
        x_tags=[]

        for j in range(len(str_i)):
            str_j=str_i[j].split('_')
            word=str_j[0]
            tag=str_j[1]
            tag=convert_to_min_tag(tag)

            if j==len(str_i)-1:
                word="."
            # print "cnt- ",i,word
            x_words.append(word)
            x_tags.append(tag)

        if i%3 ==0:
            sen_list_test_word.append(x_words)
            sen_list_test_tag.append(x_tags)
        else:
            sen_list_train_word.append(x_words)
            sen_list_train_tag.append(x_tags)

    rfile.close()
    return sen_list_train_word,sen_list_train_tag,sen_list_test_word,sen_list_test_tag,taglist

def make_sequence_window_three(model,sen_list,tag_list):
    end_PAD=model[get_unicode("</PAD>")]
    em_word=[]
    hot_vec=[]

    for sen,tags in zip(sen_list,tag_list):
        for i in range(len(sen)-1):
            three_words_em=[]
            out_tag=""

            for j in range(3):
                if i+j < len(sen):
                    three_words_em.append(model[sen[i+j]])
                else:
                    three_words_em.append(end_PAD)
                if j == 1:
                    out_tag =tags[i+j]
                pass
            three_words_em=np.reshape(three_words_em,-1)
            em_word.append(three_words_em)
            hot_vec.append(tag_hot_vec(out_tag))
        pass
    em_word=np.asarray(em_word).astype(np.float32)
    hot_vec=np.asarray(hot_vec).astype(np.float32)

    # print "final words_array shape: ",em_word.shape
    # print "final tags_array shape: ",hot_vec.shape

    return em_word,hot_vec

def make_sequence_window_one(model,sen_list,tag_list):

    end_PAD=model[get_unicode("</PAD>")]
    em_word=[]
    hot_vec=[]

    for sen,tags in zip(sen_list,tag_list):
        for i in range(len(sen)):

            # print "::------------> ",sen[i]

            one_words_em=[model[sen[i]]]
            out_tag=tags[i]
            one_words_em=np.reshape(one_words_em,-1)
            one_words_em=np.asarray(one_words_em)
            # print one_words_em.shape

            em_word.append(one_words_em)
            hot_vec.append(tag_hot_vec(out_tag))
        pass
    words_arr=np.asarray(em_word).astype(np.float32)
    tags_arr=np.asarray(hot_vec).astype(np.float32)

    # print "final words_array shape: ",words_arr.shape
    # print "final tags_array shape: ",tags_arr.shape

    return words_arr,tags_arr

def generate_em_vector(model,sen_list,tag_list,window=1):
    if window == 1:
        X,Y=make_sequence_window_one(model,sen_list,tag_list)
        return X,Y

    elif window == 3:
        X,Y=make_sequence_window_three(model,sen_list,tag_list)
        return X,Y
    else:
        print "Define Parameter Properly"
        return [],[]


def load_train(model,sen_list,tag_list,window=1):
    X,Y=generate_em_vector(model,sen_list,tag_list,window)
    return X,Y

def load_test(model,sen_list,tag_list,window=1):
    X,Y=generate_em_vector(model,sen_list,tag_list,window)
    return X,Y

def print2D(vals,tags):
    for x,y in zip(vals,tags):
        for p,q in zip(x,y):
            print get_ascii(p),get_ascii(q)

def format(val):
    return "%.17f" % val

def write_X(path,data):
    with open(path,"w") as w:
        for x in data:
            w.write(str(x[0]))
            for y in x[1:]:
                w.write(" "+str(y))
            w.write("\n")

def write_Y(path,data):
    with open(path,"w") as w:
        for x in data:
            w.write(str(x[0]))
            for y in x[1:]:
                w.write(" "+str(y))
            w.write("\n")

def load_data(data_mode="train",window=1):

    model=load_model(saved_model_path)
    sen_list_train_word,sen_list_train_tag,sen_list_test_word,sen_list_test_tag,taglist=sen_parse(data_path)

    # print2D(sen_list_train_word,sen_list_train_tag)
    # print "*"*70
    # print2D(sen_list_test_word,sen_list_test_tag)

    if data_mode=="train":
        X,Y=load_train(model,sen_list_train_word,sen_list_train_tag,window)
        write_X(network_input_train+"word_vector_0"+str(window)+".txt",X)
        write_Y(network_input_train+"tag_vector_0"+str(window)+".txt",Y)
        return X,Y

    elif data_mode=="test":
        X,Y=load_test(model,sen_list_test_word,sen_list_test_tag,window)
        write_X(network_input_test+"word_vector_0"+str(window)+".txt",X)
        write_Y(network_input_test+"tag_vector_0"+str(window)+".txt",Y)
        return X,Y
    else:
        print "----> Use: load_data(data_mode='train') or load_data(data_mode='test')"


def main():
    X,Y=load_data("train",3)

if __name__=="__main__":
     main()
