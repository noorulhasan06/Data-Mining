import numpy as np
import pandas as pd
from matplotlib import pyplot
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering

def SSE(scheme):
        totalH = 0
        for name, group in df_kmeans.groupby([scheme]):
                if group[scheme].iloc[0] != -1:
                        mean = np.c_[group['x'].mean(), group['y'].mean()]
                        data = np.c_[group['x'], group['y']]
                        result = data - mean
                        result = result*result
                        result = result.sum(axis = 1)
                        result = result.sum(axis = 0)
                        totalH += result
        return totalH


def Purity(scheme):
        totalPH = 0
        for name, group in df.groupby([scheme]):
                clus = np.array(group['class'])
                element = max(list(clus),key=list(clus).count)
                nij = (list(clus)).count(element)
                totalPH += nij
        purityH = totalPH/totalElements
        return purityH


def plotGraph(scheme, algorithm, sse, purity):
        colors={0:'red',1:'blue',2:'green',3:'orange',4:'pink',-1:'black',5:'yellow',6:'magenta',7:'brown',8:'purple'}
        pyplot.scatter(df['x'],df['y'],label=None,c=df[scheme].apply(lambda x : colors[x]))
        pyplot.xlabel('x')
        pyplot.ylabel('y')
        unique_la = set(df[scheme])
        for col in unique_la:
                if col == -1:
                        pyplot.scatter([],[],c=colors[col],label='Noise Points')
                else:
                        pyplot.scatter([],[],c=colors[col],label='clus'+str(col))
        pyplot.legend(framealpha=1,fancybox=True,loc='best')
        if algorithm != 'Original':
                pyplot.title(algorithm+' clustering')
                sse = "SSE: "+str('%.3f'%sse)
                purity = "Purity: "+str('%.3f'%purity)
                pyplot.text(26,-3,sse)
                pyplot.text(26,-4,purity)
        else :
                pyplot.title(algorithm)
        pyplot.show()



rn = np.random.randint(50,101,size=5)

df1 = pd.DataFrame(np.random.normal(loc=31 ,scale=1.9,size=[rn[0],2]),columns=['x','y'])        #LOC is the center of the distribution, scale is s.d.
df1['class'] = np.full(shape=rn[0],fill_value=0)

df2 = pd.DataFrame(np.random.normal(loc= 25,scale=2.5,size=[rn[1],2]),columns=['x','y'])
df2['class'] =  np.full(shape=rn[1],fill_value=1)

df3 = pd.DataFrame(np.random.uniform(low= 8.1,high=14.2,size=[rn[2],2]),columns=['x','y'])
df3['class'] =  np.full(shape=rn[2],fill_value=2)

df4= pd.DataFrame(np.random.exponential(scale=2.0,size=[rn[3],2]),columns=['x','y'])
df4['class'] =  np.full(shape=rn[3],fill_value=3)

df5 = pd.DataFrame(np.random.uniform(low= 7,high=30.0,size=[rn[4],2]),columns=['x','y'])
df5['class'] =  np.full(shape=rn[4],fill_value=4)

frames= [df1,df2,df3,df4,df5]
df = pd.concat(frames,ignore_index=True)
plotGraph('class','Original',0,0)
totalElements = df.shape[0]
kmeans = KMeans(n_clusters=5, random_state=0).fit(df)

df_kmeans =df
df_kmeans['clus'] = kmeans.labels_
sse_h = SSE('clus')
purity_h = Purity('clus')
plotGraph('clus','Kmeans',sse_h, purity_h)

db = DBSCAN(eps=1, min_samples=10).fit(df)
df_dbscan = df
df_dbscan['clus1'] = db.labels_
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
sse_h = SSE('clus1')
purity_h = Purity('clus1')
plotGraph('clus1','DBScan',sse_h,purity_h)

ag = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward').fit(df)
df_hier = df
df_hier['clus2']= ag.labels_
sse_h = SSE('clus2')
purity_h = Purity('clus2')
plotGraph('clus2','Hierarical',sse_h,purity_h)

