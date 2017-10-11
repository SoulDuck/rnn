import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
#Lesson 1 : how to use rnn library

def lesson_1():
    hidden_size=2
    cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
    x_data=np.array([[[1,0,0,0]]] , dtype=np.float32)
    output,_state = tf.nn.dynamic_rnn(cell,x_data,dtype=tf.float32)
    sess= tf.Session()
    sess.run(tf.global_variables_initializer())
    a=sess.run(output)
    print np.shape(a)


#Lession 2 : concept of sequence of length

def lesson_2():
    hidden_size =2
    cell = tf.contrib.rnn.BasicLSTMCell(num_units = hidden_size)
    #define sequence
    h=[1,0,0,0];e=[0,1,0,0];l=[0,0,1,0];o=[0,0,0,1]
    x_data=np.array([[h,e,l,l,o],
                     [e,l,l,o,h],
                     [l,l,o,h,e]] ,dtype=np.float32)
    print 'x data shape :', np.shape(x_data)
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=2)
    outputs, _states=tf.nn.dynamic_rnn(cell , x_data , dtype=tf.float32)
    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    a=sess.run(outputs)
    print 'output shape ' , np.shape(a)


def lesson_3(): # training
    sequence_length = 6 # i,h,e,l,l,o
    hidden_size=5
    batch_size=1
    idx2char=['h','i','e','l','o']
    x_one_hot=[[[1,0,0,0,0], # h =0
                [0,1,0,0,0], # i =1
                [0,0,1,0,0], # e =2
                [0,0,0,1,0], # l =3
                [0,0,0,1,0], # l =3
                [0,0,0,0,1]]]# o =4
    y_data=[[1,0,2,3,3,4]]
    x_=tf.placeholder(tf.float32 , [None , sequence_length , hidden_size])
    y_=tf.placeholder(tf.int32 , [None , sequence_length])

    cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_size  , state_is_tuple=True)
    initial_state=cell.zero_state(batch_size , tf.float32)
    outputs , _state = tf.nn.dynamic_rnn(cell, x_ , initial_state=initial_state , dtype = tf.float32)
    print outputs.get_shape
    print y_.get_shape
    weights=tf.ones([batch_size,sequence_length])
    sequence_loss= tf.contrib.seq2seq.sequence_loss(outputs , y_ , weights)#sequence_loss
    loss=tf.reduce_mean(sequence_loss)
    predict=tf.argmax(outputs , axis=2)
    train=tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(2000):
            l , _ =sess.run([loss, train] , feed_dict={x_:x_one_hot, y_ :y_data})
            result=sess.run(predict,feed_dict={x_: x_one_hot})
            print (i,"loss",l,"prediction",result,'true Y:',y_data)
            result_str=[ idx2char[c] for c in np.squeeze(result)]
            print ("\tPrediction str:",''.join(result_str))



def lesson_4():
    sample='if you want you'
    idx2char=list(set(sample)) #['a', ' ', 'f', 'i', 'o', 'n', 'u', 't', 'w', 'y']
    char2idx={c : i for i,c in enumerate(idx2char)}  #{'a': 0, ' ': 1, 'f': 2, 'i': 3, 'o': 4, 'n': 5, 'u': 6, 't': 7, 'w': 8, 'y': 9}
    sample_idx=[char2idx[c] for c in sample] #[3, 2, 1, 9, 4, 6, 1, 8, 0, 5, 7, 1, 9, 4, 6]
    x_data=[sample_idx[:-1]]
    y_data=[sample_idx[1:]]

    dict_size=len(char2idx)
    rnn_hidden_size=len(char2idx)
    num_classes = len(char2idx)
    batch_size=1
    sequence_length=len(sample)-1
    print len(char2idx)
    print sequence_length
    x_= tf.placeholder(tf.float32,[None , sequence_length])
    y_ = tf.placeholder(tf.int32, [None, sequence_length])
    x_onehot=tf.one_hot(x_data , depth=num_classes)

    #make cell
    cell=tf.contrib.rnn.BasicLSTMCell(num_units=rnn_hidden_size ,state_is_tuple=True)
    initial_state=cell.zero_state(batch_size,tf.float32)
    outputs , _states =tf.nn.dynamic_rnn(cell ,  x_onehot , initial_state=initial_state,dtype=tf.float32)
    #outputs , _states =tf.nn.dynamic_rnn(cell ,  x_onehot , sequence_length,initial_state=initial_state,dtype=tf.float32) --> error
    weights=tf.ones([batch_size , sequence_length] ,tf.float32)
    sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=y_, weights = weights )
    loss = tf.reduce_mean(sequence_loss)
    train=tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)
    predict =tf.argmax(outputs , axis =2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(3000):
            l,est , _ =sess.run([loss, predict , train] , feed_dict={x_:x_data , y_ : y_data })
            #print 'estimate :' , est # e.g estimate : [[2 1 9 4 6 1 8 0 5 7 1 9 4 6]]
            print i ,'\t loss:',l,'\t',''.join([idx2char[c] for i,c in enumerate(np.squeeze(est))]) #1066 	 loss: 0.979894 	f you want you


    print idx2char
    print char2idx
    print sample_idx
def lesson_5():
    ori_sentence=("if you want to build ship, don't drum up people together to"
              "collect wood and don't assign them task and work but rather"
              "teach them to long for the endless immensity of the sea")

    idx2char=list(set(ori_sentence))
    char2idx={c:i for i,c in enumerate(idx2char)}
    data_x=[]
    data_y=[]
    seq_length = 7
    for i in range(0 , len(ori_sentence) - seq_length):
        x_str = ori_sentence[i:i+seq_length]
        y_str = ori_sentence[i+1:i + seq_length+1]

        x_indices=[char2idx[c] for c in x_str]
        y_indices=[char2idx[c] for c in y_str]

        print x_indices
        print y_indices

        data_x.append(x_indices)
        data_y.append(y_indices)
    data_dim = len(char2idx)
    hidden_size = len(char2idx)
    num_classes = len(char2idx)
    batch_size = len(data_x)

    x_ = tf.placeholder(tf.int32 , [None , seq_length])
    y_ = tf.placeholder(tf.int32 , [None , seq_length])
    x_onehot = tf.one_hot(x_ , num_classes)

    cell = tf.contrib.rnn.BasicLSTMCell(hidden_size , state_is_tuple=True)
    initial_state = cell.zero_state(batch_size , tf.float32)
    outputs, states = tf.nn.dynamic_rnn(cell , x_onehot , initial_state=initial_state , dtype=tf.float32)
    weights = tf.ones([batch_size ,seq_length])
    sequence_loss=tf.contrib.seq2seq.sequence_loss(logits=outputs, targets=y_, weights=weights)
    loss=tf.reduce_mean(sequence_loss)
    train=tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
    predict=tf.argmax(outputs, axis=2)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            sequence=''
            l, _, ests=sess.run([loss, train,predict] , feed_dict={x_:data_x  ,y_: data_y})
            for j,est in enumerate(np.squeeze(ests)):
                if j==0 :
                    sequence=''.join([idx2char[t] for t in est])
                else:
                    sequence=sequence+(idx2char[est[-1]])
            print sequence

    print ori_sentence


def lesson_6():
    ori_sentence = ("if you want to build ship, don't drum up people together to"
                "collect wood and don't assign them task and work but rather"
                "teach them to long for the endless immensity of the sea")

    idx2char= list(set(ori_sentence))
    char2idx= {c:i  for i,c in enumerate(idx2char)}
    data_x=[]
    data_y=[]
    seq_length=7
    for i in range(len(ori_sentence) - seq_length):
        x_str= ori_sentence[i:i+seq_length]
        y_str= ori_sentence[i+1:i+1+seq_length]
        x_indices= [char2idx[c] for c in x_str]
        y_indices= [char2idx[c] for c in y_str]
        data_x.append(x_indices)
        data_y.append(y_indices)
    print data_x

    hidden_size=len(char2idx)
    num_classes=len(char2idx)
    batch_size=len(data_x)

    x_=tf.placeholder(tf.int32 , [None , seq_length])
    y_=tf.placeholder(tf.int32 ,  [None, seq_length])
    x_onehot=tf.one_hot(x_ , depth= num_classes)

    def lstm_cell():
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=True)
        return cell
    multi_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(2)], state_is_tuple=True)


    #initial_state=  .zero_state(batch_size , tf.float32)
    outputs , _states =tf.nn.dynamic_rnn(multi_cells , x_onehot , dtype=tf.float32 )

    x_for_softmax=tf.reshape(outputs , [-1,hidden_size])
    w_softmax=tf.get_variable('softmax_w',shape=[hidden_size ,num_classes])
    b_softmax=tf.get_variable('softmax_b',shape=[num_classes])
    outputs=tf.matmul(x_for_softmax , w_softmax)+b_softmax
    outputs=tf.reshape(outputs ,[batch_size , seq_length,num_classes])
    predict = tf.argmax(outputs , axis=2)
    weights= tf.ones([batch_size , seq_length])
    sequence_loss = tf.contrib.seq2seq.sequence_loss(logits=outputs , targets=y_ , weights = weights)
    mean_loss = tf.reduce_mean(sequence_loss)
    train = tf.train.AdamOptimizer(learning_rate=0.1).minimize(mean_loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(5000):
            sequence=''
            l, _, ests=sess.run([mean_loss, train,predict] , feed_dict={x_:data_x  ,y_: data_y})
            for j,est in enumerate(np.squeeze(ests)):
                if j==0 :
                    sequence=''.join([idx2char[t] for t in est])
                else:
                    sequence=sequence+(idx2char[est[-1]])
            print sequence
    print ori_sentence


def lesson_7():
    tf.set_random_seed(777)  # reproducibility
    if "DISPLAY" not in os.environ:
        # remove Travis CI Error
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    def MinMaxScaler(data):
        ''' Min Max Normalization
        Parameters
        ----------
        data : numpy.ndarray
            input data to be normalized
            shape: [Batch size, dimension]
        Returns
        ----------
        data : numpy.ndarry
            normalized data
            shape: [Batch size, dimension]
        References
        ----------
        .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html
        '''
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        # noise term prevents the zero division
        return numerator / (denominator + 1e-7)

    # train Parameters
    seq_length = 7
    data_dim = 5
    hidden_dim = 10
    output_dim = 1
    learning_rate = 0.01
    iterations = 500

    # Open, High, Low, Volume, Close
    xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
    xy = xy[::-1]  # reverse order (chronically ordered)
    xy = MinMaxScaler(xy)
    x = xy
    y = xy[:, [-1]]  # Close as label

    #print xy
    print np.shape(xy)
    print y[0]
    print x[0]
    print len(y)
    print len(y) - seq_length
    print x[0:seq_length]
    # build a dataset
    dataX = []
    dataY = []

    for i in range(0, len(y) - seq_length):

        _x = x[i:i + seq_length]
        _y = y[i + seq_length]  # Next close price
        print np.shape(_x)
        print np.shape(_y)
        print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)

    # train/test split
    train_size = int(len(dataY) * 0.7)
    test_size = len(dataY) - train_size
    trainX, testX = np.array(dataX[0:train_size]), np.array(
        dataX[train_size:len(dataX)])
    trainY, testY = np.array(dataY[0:train_size]), np.array(
        dataY[train_size:len(dataY)])


    print 'trainX shape', np.shape(trainX)
    print 'testX shape' , np.shape(testX)

    # input place holders
    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, 1])

    # build a LSTM network
    cell = tf.contrib.rnn.BasicLSTMCell(
        num_units=hidden_dim, state_is_tuple=True, activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
    Y_pred = tf.contrib.layers.fully_connected(
        outputs[:, -1], output_dim, activation_fn=None)  # We use the last cell's output

    # cost/loss
    loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    # RMSE
    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training step
        for i in range(iterations):
            _, step_loss = sess.run([train, loss], feed_dict={
                X: trainX, Y: trainY})
            print("[step: {}] loss: {}".format(i, step_loss))

        # Test step

        test_predict , outputs_ = sess.run([Y_pred , outputs], feed_dict={X: testX})
        rmse_val = sess.run(rmse, feed_dict={targets: testY , predictions: test_predict})
        print outputs_  , 'outputs shape' , np.shape(outputs_)
        print("RMSE: {}".format(rmse_val))
        print test_predict

        #plot predictions

        plt.plot(testY)
        plt.plot(test_predict)
        plt.xlabel("Time Period")
        plt.ylabel("Stock Price")
        plt.show()

if __name__ == '__main__':
    lesson_7()

