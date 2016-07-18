import nn_hidden_2 as nn

data_path="bd_corpus/taged_sentence.txt"
em_model_save_path="bd_save_model/model_corpus_bd"

embedding_size=100
window=1

def main():
    nn.run(window,learning_rate = 0.001,n_input=embedding_size*window)

if __name__=="__main__":
    main()
