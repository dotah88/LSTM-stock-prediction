import tushare as ts
import numpy as np
import pandas as pd
get_table=ts.get_stock_basics()
# get_table.to_csv('/home/mutong/datamining-master/tensorflow-program/rnn/stock_predict/new/csv/stockcode')
# f=open('/home/mutong/datamining-master/tensorflow-program/rnn/stock_predict/csvdata/stockcode.csv')
df=pd.read_csv(f)
codelist=np.array(df['code'])
for i in range(len(codelist)):
    codestr='%d' %codelist[i]
    alldata=ts.get_hist_data(codestr.zfill(6))

    if alldata is None:

        print(('%s%s is failed')%( i,codestr.zfill(6)))
    else:

        alldata.to_csv('/home/mutong/datamining-master/tensorflow-program/rnn/stock_predict/csvdata/'+codestr.zfill(6)+'.csv')
        print(('%s%s has saved')%(i,codestr))



# data=ts.get_hist_data('300274')
# print(data)
# data.to_csv('/home/mutong/datamining-master/tensorflow-program/rnn/stock_predict/csvdata/300274.csv')
#


