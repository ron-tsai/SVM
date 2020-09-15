from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np


#将上述过程包装成函数：
def plot_svc_decision_function(model,ax=None):
    if ax is None:
        ax=plt.gca()
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()

    x=np.linspace(xlim[0],xlim[1],30)
    y=np.linspace(ylim[0],ylim[1],30)
    y,x=np.vstack([x.ravel(),y.ravel()]).T
    P=model.decision_function(xy).reshape(x.shape)

    ax.contour(x,y,P,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
#则整个绘图过程写作

clf=SVC(kernel='linear').fit(x,y)
plt.scatter(x[:,0],x[:,1],c=y,s=50,cmap='rainbow')
plot_svc_decision_function(clf)