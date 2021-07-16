import numpy as np

def distance(p1,p2):
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


def kmeans(nums,k,tolerance=1e-4,max_iter=100):
    centers={}  #聚类中心
    
    #初始化中心点
    for i in range(k):
        centers[i]=np.random.randint(low=0,high=500,size=2)
    

    for iter in range(max_iter):
        print('this is',iter,'epoch')
        clusters={} #每个类包含那些点
        #初始化每个类的点
        for i in range(k):
            clusters[i]=[]

        #计算所有点所属的类
        for item in nums:
            all_dis=[]
            for i in range(k):
                all_dis.append(distance(item,centers[i]))
            clusters[np.argmin(all_dis)].append(item)


        #计算新的聚类中心
        pre_centers=centers.copy()
        for i in range(k):
            centers[i]=np.mean(clusters[i],axis=0)

        
        #判断是否还需要继续更新
        ok=True
        for i in range(k):
            if(abs(np.sum((centers[i]-pre_centers[i])**2))>tolerance):
                ok=False
                break
        if(ok==True):
            break

    return centers,clusters
        

input=np.random.randint(low=0,high=500,size=(500,2))
center,cluster=kmeans(input,k=4)
print(center)
        
    
import matplotlib.pyplot as plt

colors=['r','g','b','y']

for k,points in cluster.items():
    for p in points:
        plt.scatter(p[0],p[1],color=colors[k])

for k,point in center.items():
    plt.scatter(point[0],point[1],color=colors[k],marker='*')

plt.show()