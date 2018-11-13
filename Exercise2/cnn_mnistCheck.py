from __future__ import print_function

import argparse
import gzip
import json
import os
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def mnist(datasets_dir='./data'):
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 28, 28, 1)
    test_y = test_y.astype('int32')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 28, 28, 1)
    valid_y = valid_y.astype('int32')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 28, 28, 1)
    train_y = train_y.astype('int32')
    print('... done loading data')
    return train_x, one_hot(train_y), valid_x, one_hot(valid_y), test_x, one_hot(test_y)


def train_and_validate(x_train, y_train, x_valid, y_valid, x_test, y_test, num_epochs, lr, num_filters, batch_size, filtersize):
    dev = '/GPU:0'
    tf.device(dev)
    tf.reset_default_graph()
    #learning curve 
    learning_curve = []
    # Input layer 
    x = tf.placeholder(dtype=tf.float32)
    input_layer = tf.reshape(x, [-1, 28, 28, 1])

    y = tf.placeholder(tf.float32, [None, 10])
    # Convolutional Layer 1 
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=num_filters, kernel_size=[filtersize,filtersize], padding="SAME", activation=tf.nn.relu)
    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1, padding="SAME")
    # Convolutional Layer 2
    conv2 = tf.layers.conv2d(inputs = pool1, 
    filters = num_filters, 
    kernel_size=[filtersize,filtersize], 
    padding="SAME", 
    activation=tf.nn.relu)

    # pooling layer 2 
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=1, padding="SAME")

    # fully connected layer 
    tmp = tf.reshape(pool2, [-1, 28 * 28 * num_filters])
    fully = tf.layers.dense(inputs=tmp, units=128, activation=tf.nn.relu)

    # softmax 
    softmax = tf.layers.dense(inputs=fully, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=softmax, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(softmax, name="softmax_tensor")
    }

     # loss 
    loss = lossFunc(y, softmax)
    

    # train model 
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    trainOp = optimizer.minimize(loss=loss)

    # evaluation metric
    eval_metric_ops=accuracy(y, softmax)

	#setup the initialisation operator
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
    # initialise the variables
        sess.run(init_op)
        total_batch = int(len(y_train) / batch_size)
        total_batch2 = int(len(y_valid) / batch_size)
        for epoch in range(num_epochs):
            avg_cost = 0
            avg_acc = 0
	    
            for i in range(total_batch):
                batch_x = x_train[(i *batch_size):((i+1)*batch_size)]
                batch_y = y_train[(i *batch_size):((i+1)*batch_size)]
                _, c = sess.run([trainOp, loss], feed_dict={x: batch_x, y: batch_y})
                avg_cost += c / total_batch
            for j in range(total_batch2):
                batch_x2 = x_valid[(j*batch_size):((j+1)*batch_size)]
                batch_y2 = y_valid[(j*batch_size):((j+1)*batch_size)]
                vali_acc = sess.run(eval_metric_ops, feed_dict={x: batch_x2, y:batch_y2 })
                avg_acc += vali_acc/total_batch2
            learning_curve.append(avg_cost)
            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost), "validation  accuracy: {:.3f}".format(avg_acc))

        print("\nTraining complete!")
        total_batch3 = int(len(y_test)/batch_size)
        avg_error = 0
        for k in range(total_batch3):
            batch_x3 = x_test[(k*batch_size):((k+1)*batch_size)]
            batch_y3 = y_test[(k*batch_size): ((k+1)*batch_size)]
            test_accu = sess.run(eval_metric_ops, feed_dict={x: batch_x3, y:batch_y3})
            avg_error += test_accu / total_batch3
        print("{:.3f}".format(1-avg_error))
       
    return learning_curve, softmax, 1-avg_error  

def lossFunc(y, y_): 
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
    loss = tf.reduce_mean(loss)
    tf.summary.scalar('cross_entropy', loss)
    return loss

def accuracy(y, y_):
    pred = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    pred = tf.cast(pred, tf.float32)
    eval_metric_ops=tf.reduce_mean(pred)
    return eval_metric_ops



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", default="./", type=str, nargs="?",
                        help="Path where the results will be stored")
    parser.add_argument("--input_path", default="./", type=str, nargs="?",
                        help="Path where the data is located. If the data is not available it will be downloaded first")
    parser.add_argument("--learning_rate", default=0.0001, type=float, nargs="?", help="Learning rate for SGD")
    parser.add_argument("--num_filters", default=16, type=int, nargs="?",
                        help="The number of filters for each convolution layer")
    parser.add_argument("--batch_size", default=128, type=int, nargs="?", help="Batch size for SGD")
    parser.add_argument("--epochs", default=12, type=int, nargs="?",
                        help="Determines how many epochs the network will be trained")
    parser.add_argument("--run_id", default=8, type=int, nargs="?",
                        help="Helps to identify different runs of an experiments")

    args = parser.parse_args()

    # hyperparameters
    lr = args.learning_rate
    num_filters = args.num_filters
    batch_size = args.batch_size
    epochs = args.epochs
    filtersize = 7

    print("mycoode")
	#Load trianing data 
    train_x, train_y, valid_x, valid_y, test_x, test_y = mnist(args.input_path)

    # set up logging 
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)


    # train and test convolutional neural network
    learning_curve, model, test_error = train_and_validate(train_x, train_y, valid_x, valid_y, test_x, test_y, epochs, lr, num_filters, batch_size, filtersize)


    # save results in a dictionary and write them into a .json file
    results = dict()
    results["lr"] = lr
    results["num_filters"] = num_filters
    results["batch_size"] = batch_size
    results["learning_curve"] = learning_curve
    results["test_error"] = test_error

    path = os.path.join(args.output_path, "results")
    os.makedirs(path, exist_ok=True)

    fname = os.path.join(path, "results_run_%d.json" % args.run_id)

    fh = open(fname, "w")
    json.dump(results, fh)
    fh.close()

