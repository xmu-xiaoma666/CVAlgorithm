import numpy as np
import random
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import size

#求两个点的欧氏距离
def distance(x,y):
    return np.sqrt(np.sum((x-y)**2))


def kmean(data,tolerance=0.0001,k=2,max_iter=500):
    centers={} #聚类中心
    #初始化聚类中心
    for i in range(k):
        centers[i]=np.random.randint(low=1,high=50,size=2)

    #开始训练
    for i in range(max_iter):
        print('这是第',i,'次迭代')
        clusters={} #聚类结果 {聚类下标：[该类的样本1，该类的样本2......]}
        #初始化每类的结果为空
        for i in range(k):
            clusters[i]=[]
        
        #遍历每个样本：计算该样本跟每个聚类中心的距离，分为离聚类中心最近的类
        for sample in data:
            distances=[]
            #计算样本点和每个聚类中心的距离
            for i in range(k):
                tmp_dis=distance(sample,centers[i])
                distances.append(tmp_dis)
            #求出最近的聚类中心，就是他的类别
            cls_idx=np.argmin(distances)
            #加入到该类的样本中
            clusters[cls_idx].append(sample)

        pre_centers = centers.copy()  # 记录之前的聚类中心点
        
        #计算新的聚类中心
        for i in range(k):
            centers[i]=np.mean(clusters[i],axis=0)
        
        #判断是否继续优化(判断和上一次的变化)
        ok=True
        for i in range(k):
            if(abs(np.sum((centers[i]-pre_centers[i])**2))>tolerance):
                ok=False
                break
        if(ok):
            break
    
    return centers,clusters



x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
x = np.random.randint(1,high=100,size=(50,2))
center,cluster=kmean(x)
print(center)

#可视化
color=['r','g','b']
for k,v in cluster.items():
    for point in v:
        plt.scatter(point[0],point[1],c=color[k])

for k,v in center.items():
    print(k,v)
    plt.scatter(v[0],v[1],c=color[k],marker='*')

plt.show()