import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import read_train_data as re
import read_test_data as re1
import random



#定义常量
rnn_unit=256     #hidden layer units
input_size=14
lr=0.006  #学习率
class_num=4
batch_size=32
train_num=10000
keep_prob = tf.placeholder(tf.float32)

#——————————————————导入数据——————————————————————
def get_train_data(batch_size=32,time_step=20):
    train_x=[]
    train_y=[]
    for i in range(batch_size) :
        a=random.randint(0,len(re.data_train)-1)
        train_data=re.data_train[a]                     #随机取一支股票
        b=random.randint(0,len(train_data)-time_step-1)
        train_x.append(train_data[b:b+time_step])       #随机取20天
        train_label=re.data_label[a]
        train_y.append(train_label[b+time_step-1])
    return train_x,train_y


def get_test_data(batch_size=32,time_step=20):
    test_x=[]
    test_y=[]
    for i in range(batch_size) :
        a=random.randint(0,len(re1.data_test)-1)
        test_data=re1.data_test[a]
        b=random.randint(0,len(test_data)-time_step-1)
        test_x.append(test_data[b:b+time_step])
        test_label=re1.data_label[a]
        test_y.append(test_label[b+time_step-1])
    return test_x,test_y


# # #——————————————————定义神经网络变量——————————————————
# #输入层、输出层权重、偏置
#
weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,7]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[7,]))
       }
#
#——————————————————定义神经网络变量——————————————————
def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #需要将tensor转成2维进行计算，计算后的结果作为隐藏层的输入
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #将tensor转成3维，作为lstm cell的输入
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit,forget_bias=1.0,state_is_tuple=True)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
        output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    h_state=output_rnn[:,-1,:]
    W = tf.Variable(tf.truncated_normal([rnn_unit, class_num], stddev=0.1), dtype=tf.float32)
    bias = tf.Variable(tf.constant(0.1, shape=[class_num]), dtype=tf.float32)
    y_pre = tf.nn.softmax(tf.matmul(h_state, W) + bias)
    return y_pre,final_states


# #——————————————————训练模型——————————————————
def train_lstm(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,class_num])
    y_pre,_=lstm(X)
    cross_entropy = -tf.reduce_mean(Y * tf.log(y_pre))
    train_op = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    #module_file = tf.train.latest_checkpoint()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, module_file)
        #重复训练10000次
        for i in range(train_num):
            train_x,train_y=get_train_data()
            loss_,train_accuracy=sess.run([cross_entropy,accuracy], feed_dict={X:train_x,Y:train_y, keep_prob: 1.0})
            print(i,loss_,train_accuracy)
            sess.run(train_op, feed_dict={X:train_x,Y:train_y, keep_prob: 0.8})
            if i % 100==0:
                print("保存模型：",saver.save(sess,'./save/stock2.model',global_step=i))

train_lstm()

def prediction(time_step=20):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,class_num])
    y_pre, _ = lstm(X)
    # mean,std,test_x,test_y=get_test_data(time_step,test_begin)
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('./save/')
        saver.restore(sess, module_file)
        for i in range(10):
            test_x, test_y = get_test_data()
            print("test accuracy %g" % sess.run(accuracy, feed_dict={X:test_x,Y:test_y}))
            pred=sess.run(y_pre,feed_dict={X:test_x,Y:test_y})




prediction()

