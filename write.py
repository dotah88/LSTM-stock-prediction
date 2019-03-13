import pandas as pd
import numpy as np
import os
from tqdm import tqdm


filespath = "./csvdata/"
output_file = "./bin/train_data.bin"
write_line = 0
files = os.listdir(filespath)
filename=[]
filelines=[]
alldata=[]
for file in tqdm(files):
    filepath = os.path.join(filespath, file)
    df = pd.read_csv(filepath)
    data=df.iloc[:,1:15].values  #取第3-10列
    data=data[::-1]


    filename.append(file)
    alldata=np.vstack((alldata,data))
    filelines.append(data.shape[0])

    write_line = write_line + data.shape[0]
    with open(output_file, 'a') as fout:
        alldata.tofile(fout)
        print('write total ' + str(write_line) + ' lines')

