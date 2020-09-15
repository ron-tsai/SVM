from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

x,y = make_blobs(n_samples=50,centers=2,random_state=0,cluster_std=0.6)
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='rainbow')
ax=plt.gca()
#获取平面上两条坐标轴最大值和最小值
xlim=ax.get_xlim()
ylim=ax.get_ylim()
#在最大值和最小值之间形成30个规律数据
axisx=np.linspace(xlim[0],ylim[1],30)
axisy=np.linspace(ylim[0],ylim[1],30)
axisy,axisx=np.meshgrid(axisy,axisx)
#我们将使用这里形成的二维数组作为我们contour函数中的x和y
#使用meshgrid函数将两个一维向量转换为特征矩阵
#核心是将两个特征向量广播，以便获取y.shape* x.shape这么多坐标点的横坐标和纵坐标
xy=np.vstack([axisx.ravel(),axisy.ravel()]).T
#其中ravel()是降维函数，vstack能够将多个结构一致的一维数组按行堆叠起来
#xy就是已经形成的网格，它是遍布在整个画布上的密集的点
plt.scatter(xy[:,0],xy[:,1],s=1,cmap='rainbow')
#理解函数meshgrid和vstack作用
a=np.array([1,2,3])
b=np.array([7,8])
#两两组合会得到多少个坐标？
#答案是六个，分别是（1,7），（2,7），（3,7），（1,8），（2,8），（3,8）
v1,v2=np.meshgrid(a,b)
v1
v2
v=np.vstack([v1.ravel(),v2.ravel()]).T
plt.xticks([])
plt.yticks([])
plt.show()





