import numpy as np
import struct

test_data_file = "./bin/test_data.bin"
test_label_file="./bin/test_label.bin"

# 62933
filename=['000524.csv', '000521.csv']
fileline=[619, 713]
s1 = open(test_data_file, 'rb')
s2 = open(test_label_file, 'rb')
train_data_temp = s1.read(1332*14*4)
train_label_temp = s2.read(1332*4)

d=[]
l=[]
label=[]


for i in range(0,1332*14*4,4):
    a = train_data_temp[i: i + 4]
    b = struct.unpack("f", a)
    # print(b)
    c = b[0]
    d.append(c)
e=np.array(d).astype(np.float32)
f=np.reshape(e,[-1,14])

for i in range(0,1332*4,4):
    a = train_label_temp[i: i + 4]
    b = struct.unpack("f", a)
    c = b[0]
    l.append(c)
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

head=0
first_data=f[0:fileline[0]]
first_label=label[0:fileline[0]]
data_test=[]
data_label=[]
data_test.append(first_data)
data_label.append(first_label)
for i in range(2):
    if i ==0:
        test_data=f[0:fileline[i]]
        test_label = f[0:fileline[i]]
    else:
        head = head + fileline[i-1]
        tail=head+fileline[i]
        test_data=f[head:tail]
        test_label=label[head:tail]
        data_test.append(test_data)
        data_label.append(test_label)
s1.close()
s2.close()

