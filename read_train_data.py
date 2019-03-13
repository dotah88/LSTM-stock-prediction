
import numpy as np
import struct
train_data_file = "./bin/train_data.bin"
train_label_file="./bin/train_label.bin"

# 62933
filename=['000516.csv', '000069.csv', '000046.csv', '000301.csv', '000023.csv', '000514.csv', '000014.csv', '000407.csv', '000153.csv', '000007.csv', '000422.csv', '000425.csv', '000010.csv', '000418.csv', '000090.csv', '000028.csv', '000505.csv', '000066.csv', '000005.csv', '000030.csv', '000166.csv', '000155.csv', '000517.csv', '000401.csv', '000004.csv', '000099.csv', '000088.csv', '000159.csv', '000419.csv', '000021.csv', '000420.csv', '000488.csv', '000070.csv', '000036.csv', '000502.csv', '000400.csv', '000409.csv', '000065.csv', '000002.csv', '000411.csv', '000034.csv', '000519.csv', '000001.csv', '000049.csv', '000150.csv', '000058.csv', '000060.csv', '000503.csv', '000100.csv', '000430.csv', '000019.csv', '000408.csv', '000026.csv', '000089.csv', '000428.csv', '000035.csv', '000423.csv', '000039.csv', '000416.csv', '000158.csv', '000050.csv', '000062.csv', '000038.csv', '000078.csv', '000037.csv', '000511.csv', '000016.csv', '000017.csv', '000068.csv', '000011.csv', '000415.csv', '000045.csv', '000055.csv', '000012.csv', '000056.csv', '000501.csv', '000032.csv', '000402.csv', '000063.csv', '000008.csv', '000404.csv', '000520.csv', '000043.csv', '000006.csv', '000429.csv', '000507.csv', '000009.csv', '000403.csv', '000498.csv', '000048.csv', '000518.csv', '000027.csv', '000506.csv', '000156.csv', '000151.csv', '000417.csv', '000040.csv', '000031.csv']
fileline=[557, 725, 668, 580, 542, 721, 731, 633, 476, 246, 704, 715, 712, 731, 724, 620, 603, 534, 714, 723, 701, 315, 699, 617, 558, 608, 731, 619, 731, 711, 726, 725, 718, 729, 637, 731, 585, 628, 597, 694, 337, 591, 732, 730, 605, 645, 725, 538, 568, 705, 580, 701, 730, 729, 730, 479, 731, 731, 626, 725, 598, 693, 366, 728, 615, 254, 646, 699, 682, 732, 465, 643, 725, 731, 618, 731, 588, 731, 686, 658, 722, 567, 662, 615, 671, 723, 701, 526, 702, 599, 670, 731, 525, 646, 729, 727, 621, 550]

s1 = open(train_data_file, 'rb')
s2 = open(train_label_file, 'rb')
train_data_temp = s1.read(62933*14*4)
train_label_temp = s2.read(62933*4)

d=[]
l=[]
label=[]
#1669980

for i in range(0,62933*14*4,4):
    a = train_data_temp[i: i + 4]
    b = struct.unpack("f", a)
    # print(b)
    c = b[0]
    d.append(c)
e=np.array(d).astype(np.float32)
f=np.reshape(e,[-1,14])

for i in range(0,62933*4,4):
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
data_train=[]
data_label=[]
data_train.append(first_data)
data_label.append(first_label)
for i in range(98):
    if i ==0:
        train_data=f[0:fileline[i]]
        train_label = label[0:fileline[i]]
    else:
        head = head + fileline[i-1]
        tail=head+fileline[i]
        train_data=f[head:tail]
        train_label=label[head:tail]
        data_train.append(train_data)
        data_label.append(train_label)
s1.close()
s2.close()




