from os import scandir
import numpy as np 




def nms(dets,threod):
    x1=dets[:,0]
    y1=dets[:,1]
    x2=dets[:,2]
    y2=dets[:,3]
    scores=dets[:,4]
    area=(x2-x1+1)*(y2-y1+1) #计算所有框的面积
    keep=[]
    order=scores.argsort()[::-1]

    while len(list(order)) >0:
        i=order[0]
        keep.append(i)

        xx1=np.maximum(x1[i],x1[order[1:]])
        yy1=np.maximum(y1[i],y1[order[1:]])
        xx2=np.minimum(x2[i],x2[order[1:]])
        yy2=np.minimum(y2[i],y2[order[1:]])

        h=np.maximum(yy2-yy1+1,0)
        w=np.maximum(xx2-xx1+1,0)
        inter=h*w
        iou=inter/(area[i]+area[order[1:]]-inter)
        idx=np.where(iou<=threod)[0]
        order=order[idx+1]
    return keep



boxes=np.array([[0,0,2,2,0.1],[0,1,2,2,0.3],[1,0,2,2,0.5],[1,1,3,2,0.6]])
keep=nms(boxes,0.4)
print(boxes[keep])