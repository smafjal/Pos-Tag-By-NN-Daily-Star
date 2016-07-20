import nn_hidden_2 as nn

embedding_size=100
window=1

def main():
    nn.run(window,learning_rate = 0.001,n_input=embedding_size*window)

if __name__=="__main__":
    main()
