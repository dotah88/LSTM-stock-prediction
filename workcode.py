import tushare as ts
import numpy as np
import pandas as pd

f = open('/home/mutong/datamining-master/tensorflow-program/rnn/stock_predict/csvdata/stockcode.csv')
df = pd.read_csv(f)
codelist = np.array(df['code'])
def get_work_code():
    for i in range(len(codelist)):
        codestr = '%d' % codelist[i]
        alldata = ts.get_hist_data(codestr.zfill(6))
        if alldata is None:
                continue
        else:
            work_code=codestr.zfill(6)
            print(('%s %s is training') % (i, work_code))
    return work_code


    # data=ts.get_hist_data('300274')
# print(data)
# data.to_csv('/home/mutong/datamining-master/tensorflow-program/rnn/stock_predict/csvdata/300274.csv')