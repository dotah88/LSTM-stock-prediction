import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


#定义常量
time_step=10
rnn_unit=256     #hidden layer units
input_size=14
lr=0.006  #学习率
class_num=4
keep_prob = tf.placeholder(tf.float32)

#——————————————————导入数据——————————————————————



f=open('/home/mt/代码/000001.csv')
df=pd.read_csv(f)
l=np.array(df['label'])
l=l[::-1]
data=df.iloc[:,2:16].values
train_data=data[::-1]
train_data=(train_data-np.mean(train_data,axis=0))/np.std(train_data,axis=0)
label=[]
for i in l:
    if i<=-0.05:
        y=[1,0,0,0]
        label.append(y)
    elif -0.05<i<=0:
        y=[0,1,0,0]
        label.append(y)
    elif 0<i<= 0.05:
        y=[0,0,1,0]
        label.append(y)
    elif i>0.05:
        y=[0,0,0,1]
        label.append(y)

x = []
y = []
for i in range(len(train_data)-time_step):
    x1 = train_data[i:i + time_step]
    x.append(x1)
    y1 = label[i + time_step-1]
    y.append(y1)
y=np.reshape(y,[722,4])



#输入层、输出层权重、偏置

weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }

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
def train_lstm(batch_size=64,time_step=10,train_begin=0,train_end=5800):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,class_num])
    # batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    y_pre,_=lstm(X)

    #损失函数
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

        for i in range(11):
            step = 0
            start = 0
            end = start + batch_size
            while (end < len(x)):
                loss_,train_accuracy = sess.run([cross_entropy,accuracy], feed_dict={X: x[start:end], Y: y[start:end], keep_prob: 1.0})
                print(i, loss_,train_accuracy)
                sess.run(train_op, feed_dict={X: x[start:end], Y:y[start:end], keep_prob: 0.8})
                pred = sess.run(y_pre, feed_dict={X: x[start:end], Y:y[start:end], keep_prob: 0.8})
                start += batch_size
                end = start + batch_size
                if i % 10==0:
                    print("保存模型：",saver.save(sess,'./save/stock_soft.model',global_step=i))
                # plt.draw()
        # for i in range(21):
        #     for step in range(len(batch_index)-1):
        #         loss_,train_accuracy=sess.run([cross_entropy,accuracy], feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]], keep_prob: 1.0})
        #     print(i,loss_,train_accuracy)
        #     sess.run(train_op, feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]], keep_prob: 0.5})
        #     if i % 10==0:
        #         print("保存模型：",saver.save(sess,'./save/stock2.model',global_step=i))

train_lstm()



def prediction(time_step=10):
    test_x=x
    test_y=y
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,class_num])
    y_pre, _ = lstm(X)

    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        #参数恢复
        module_file = tf.train.latest_checkpoint('./save')
        saver.restore(sess, module_file)
        # for i in range(len(test_x)-time_step):
        print("test accuracy %g" % sess.run(accuracy, feed_dict={X:test_x,Y:test_y}))
        pred=sess.run(y_pre,feed_dict={X:test_x,Y:test_y})
        # plt.figure()
        # plt.plot(list(range(len(test_y[0:20]))), pred[0:20], color='b')
        # plt.plot(list(range(len(y[0:20]))), y[0:20],  color='r')
        # plt.show()










