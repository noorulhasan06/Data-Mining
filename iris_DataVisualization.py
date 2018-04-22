import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

my_list = [[1,2,3,4,5,],['jan','feb','mar'],np.matrix([[1,2,3],[4,5,6]])]
df1 = pd.DataFrame({
      'letters':['a','b','c','d','e'],
      'norm_dist_nums':np.random.normal(size=5),
      'nums':(1,2,3,4,5),
      'logical':np.random.uniform(size=5)>0
    })

df2 = pd.DataFrame({
        'letters':['f','g','c','d','e'],
        'norm_dist_nums':np.random.normal(size=5),
        'nums':(6,7,3,4,5),
        'logical':np.random.uniform(size=5)>0
  })

df3 = pd.concat([df1,df2],ignore_index=True)
df4 = pd.concat([df1,df2],axis=1)

df5 = pd.merge(df1,df2,how='inner',on='letters')
df6 = pd.merge(df1,df2,how='outer',on='letters')

colSum = (df1.sum(axis = 0))[2:3]
colMean = (df1.mean(axis = 0))[1:2]

names = ['sepal_length','sepal_width','petal_length','petal_width','class']
iris = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',names= names)

sepal_length = pd.DataFrame(iris['sepal_length'],columns=['sepal_length'])

plt.rcParams['patch.edgecolor']='g'
plt.rcParams['patch.force_edgecolor']=True

sepal_length.plot(kind='hist',label='sepal_length',bins=10)
plt.show()

iris.plot(kind='scatter',x='sepal_length',y='sepal_width')
plt.show()
