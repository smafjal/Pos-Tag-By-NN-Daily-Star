#!/usr/bin/env python

r_1="train.txt"
w_1="/home/afjal/Github/Pos-Tag-By-NN-Daily-Star/input_tt/trainParsed.txt"
only_w_1="/home/afjal/Github/Pos-Tag-By-NN-Daily-Star/daily_star_corpus/trainParsedSen.txt"

def process_sen(sen):
    y=" ".join([x for x in sen])
    return y.strip()

def parse_data(path):

    sentences=[]
    with open(path,"r") as r:
        local_sen=[]
        for x in r:
            if not x or x.isspace():
                continue
            x_split=x.strip().split()
            word=x_split[0]
            tag=x_split[1]

            if word=="," or word=="#" or word=="$" or word==")" or word=="(" or word=="-RRB-" or word=="-LRB-" or  word=="-LCB-" or  word=="-RCB-" or word==":" or word=="," or word == "``" or word == "\"\"" or word == ";" or word == "--" or word == "?":
                continue
            if tag=="," or tag=="#" or tag=="$" or tag==")" or tag=="(" or tag=="-RRB-" or tag=="-LRB-" or  tag=="-LCB-" or  tag=="-RCB-" or tag==":" or tag=="," or tag == "``" or tag == "\"\"" or tag == ";" or tag == "--" or tag == "?" or tag== "''" or (tag=="." and word != "."):
                continue

            if word==".":
                sentences.append(process_sen(local_sen))
                local_sen=[]
            else:
                local_sen.append(word)
                local_sen.append(tag)

    return sentences


def print_list(lis):
    for i in range(min(10,len(lis))):
        print lis[i]

def write_sen(sentences,path):
    with open(path,"w") as w:
        for x in sentences:
            w.write(x+"\n")

def corpus_make(path):
    sentences=[]
    with open(path,"r") as r:
        for x in r:
            local_sen=""
            x=x.strip().split()
            for i in range(0,len(x),2):
                word=x[i]
                tag=x[i+1]
                local_sen=local_sen+" "+word
            sentences.append(local_sen.strip())
    return sentences

def main():
    print "Parse File"
    sentences=parse_data(r_1)
    print "sen-len: ",len(sentences)
    print_list(sentences)
    write_sen(sentences,w_1)

    print "*"*60
    print "Corpus making"
    sen_only=corpus_make(w_1)
    write_sen(sen_only,only_w_1)

if __name__=="__main__":
    main()

