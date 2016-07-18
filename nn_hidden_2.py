import tensorflow as tf
import numpy as np
import  generate_word_em_vector as data_reader

taglist=['N','V','J','D','L','A','C','RD','P','PU','PP']

def neuralNetwork(window=1,learning_rate = 0.001,n_input=100):

    print "-----------> Loading Data files <---------------"
    train_word,train_tag=data_reader.load_data("train",window)
    test_word,test_tag=data_reader.load_data("test",window)

    print "\n"
    print "-- Train Shape -- "
    print train_word.shape
    print train_tag.shape

    print "-- Test Shape -- "
    print test_word.shape
    print test_tag.shape

    # Parameters
    batch_size = 1500
    display_step = 1
    training_epochs=50

    # Network Parameters
    # n_input = 100 # embading size
    n_classes = len(taglist)
    n_hidden_1 = int((n_input+n_classes)/2) + 10 # 1st layer num features
    n_hidden_2 = int((n_input+n_classes)/2) + 10 # 2nd layer num features

    print "net-input-size: ",n_input
    print "net-output-size: ",len(taglist)
    print "net-hidden-1 size: ",n_hidden_1
    print "net-hidden-2 size: ",n_hidden_2
    print "*"*60

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    def multilayer_perceptron(_X, _weights, _biases):
        layer_1 = tf.nn.relu(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) #Hidden layer with RELU activation
        layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, _weights['h2']), _biases['b2'])) #Hidden layer with RELU activation
        return tf.matmul(layer_2, _weights['out']) + _biases['out']

    pred = multilayer_perceptron(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) # Softmax loss
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

    num_examples=len(train_word)


    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(num_examples/batch_size)
            start_idx=0;

            for i in range(total_batch):
                st_idx=start_idx
                ed_idx=min(st_idx+batch_size,num_examples)

                batch_xs, batch_ys = train_word[st_idx:ed_idx],train_tag[st_idx:ed_idx]
                sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
                avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/num_examples

            if epoch % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost)

        print "-------->> optimization Finished <<-----------"
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        test_len=100
        itr_step=0
        total_ac=0;

        for i in range(min(100,len(test_word)/test_len)):
            st_idx=i*test_len
            end_idx=min(st_idx+test_len,len(test_word))
            accuracy_val=accuracy.eval({x: test_word[st_idx:end_idx], y: test_tag[st_idx:end_idx]})
            total_ac=total_ac+accuracy_val
            itr_step=itr_step+1
            print "iterate: ",i," accuracy-val: ",accuracy_val

        print "Final - Accuracy: ","{:.9f}".format(total_ac/itr_step)
	accuracy_all_data=accuracy.eval({x:test_word,y:test_tag}) 
	print "Accuracy All Data: ","{:.9f}".format(accuracy_all_data)
	

def run(window=1,learning_rate = 0.001,n_input=100):
    neuralNetwork(window,learning_rate,n_input)

def main():
    run(1,0,001,100)

if __name__=="__main__":
    main()

