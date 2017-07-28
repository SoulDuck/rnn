import tensorflow as tf
import numpy as np

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
if __name__ == '__main__':
    lesson_4()