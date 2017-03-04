import tensorflow as tf
import DataGetter
import numpy as np

def main(x_train,y_train,x_test,y_test):
    
    n_labels = np.unique(y_train).size
    
    INPUT_SIZE = 561
    OUTPUT_SIZE = n_labels
    HIDDEN_SIZES = [512,256,128,64,32,16]
    NUM_TRAINING_STEPS = 2000
    
    y_train = DataGetter.reformat(y_train,n_labels)
    y_test = DataGetter.reformat(y_test,n_labels)
    

    keep_prob = tf.placeholder(tf.float32) 
    
    initializer = tf.random_normal_initializer(stddev=0.1)
    
    # Setup Placeholders => None argument in shape lets us pass in arbitrary sized batches
    X = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])  
    Y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])
    
    # Hidden Layer Variables
    
    inp = X
    inp_size = INPUT_SIZE
    h_size = HIDDEN_SIZES[0]
    
    for i in range(0,len(HIDDEN_SIZES)):
        print("creating layer: " + str(i))
        wtmp = tf.get_variable("Hidden_W%d" %(i), shape=[inp_size, h_size], initializer=initializer)
        btmp = tf.get_variable("Hidden_b%d" %(i), shape=[h_size], initializer=initializer)
    
        # Hidden Layer Transformation
        hidden = tf.nn.relu(tf.matmul(inp, wtmp) + btmp)
        hidden_drop = tf.nn.dropout(hidden, keep_prob)
    
        inp_size = h_size
        inp = hidden_drop
        
        if(i < len(HIDDEN_SIZES)-1):
            h_size = HIDDEN_SIZES[i+1]
    
    # Output Layer Transformation
    w_out = tf.get_variable("Output_W", shape=[HIDDEN_SIZES[-1], OUTPUT_SIZE], initializer=initializer)
    b_out = tf.get_variable("Output_b", shape=[OUTPUT_SIZE], initializer=initializer)
    output = tf.matmul(inp, w_out) + b_out
    
    # Compute Loss
    loss = tf.losses.softmax_cross_entropy(Y, output)
    
    # Compute Accuracy
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(output, 1))
    accuracy = 100 * tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # Setup Optimizer
    train_op = tf.train.AdamOptimizer().minimize(loss)
    
    ### Launch the Session, to Communicate with Computation Graph ###
    with tf.Session() as sess:
        # Initialize all variables in the graph
        sess.run(tf.global_variables_initializer())
    
        # Training Loop
        for i in range(NUM_TRAINING_STEPS):
            curr_acc, _ = sess.run([accuracy, train_op], feed_dict={X: x_train, Y: y_train, keep_prob: 0.5})
            if i % 100 == 0:
                print('Step %d Current Training Accuracy: %.3f' % (i, curr_acc))
        
        # Evaluate on Test Data
        print('Test Accuracy: %.3f' % sess.run(accuracy, feed_dict={X: x_test, 
                                                                    Y: y_test, keep_prob: 1.0}))
