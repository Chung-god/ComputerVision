import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.svm import SVC
from sklearn import svm
from mpl_toolkits import mplot3d
from sklearn.gaussian_process.kernels import RBF


#---------------------------------------------------------------------------------------
# Part1.Linear SVM
# 1. Create Datasets
#---------------------------------------------------------------------------------------
#make_blobs
'''
n_samples : 표본 데이터의 수, 디폴트 100
centers : 생성할 클러스터의 수 혹은 중심, [n_centers, n_features] 크기의 배열. 디폴트 3
cluster_std: 클러스터의 표준 편차, 디폴트 1.0
random_state : 난수 발생 시드

반환값:
   X : [n_samples, n_features] 크기의 배열 (독립변수)
   y : [n_samples] 크기의 배열 (종속변수)
'''
#random_state = 20
X,y = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=20)
plt.scatter(X[:,0], X[:,1], c=y, s=30)
plt.title('datasets random_state=20')
# plt.show()
# plt.savefig('./img/datesets20.png')
plt.clf()


X,y = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=30)
plt.scatter(X[:,0], X[:,1], c=y, s=30)
plt.title('datasets random_state=30')
# plt.show()
# plt.savefig('./img/datesets30.png')
plt.clf()

X,y = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=40)
plt.scatter(X[:,0], X[:,1], c=y, s=30)
plt.title('datasets random_state=40')
# plt.show()
# plt.savefig('./img/datesets40.png')
plt.clf()

#---------------------------------------------------------------------------------------
# 2. Train SVM
#---------------------------------------------------------------------------------------
X,y = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=50)

#fit the modle
clf = svm.SVC(kernel='linear',C=1.0)
clf.fit(X,y)

plt.scatter(X[:,0], X[:,1], c=y, s=30, cmap=plt.cm.Paired)

# plot the decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])

# plot support vectors
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=100, linewidth=1, facecolors='none', edgecolors='k')
plt.title('C=1.0')
# plt.savefig('./img/SVM1.png')
# plt.show()
plt.clf()

#---------------------------------------------------------------------------------------
# Part2. Nonliear SVM
# 1. Create Datasets                                                                                   
#---------------------------------------------------------------------------------------
X,y=make_circles(factor=0.1,noise=0.1) #factor = R2/R1, noise=std
plt.scatter(X[:,0], X[:,1],c=y, s=30, cmap=plt.cm.Paired)
plt.title('factor=0.1')
# plt.savefig('./img/datasetsNL.png')
# plt.show()
plt.clf()

#---------------------------------------------------------------------------------------
# 2. Kernel function
#---------------------------------------------------------------------------------------

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

X,y=make_circles(factor=0.1,noise=0.1) #factor = R2/R1, noise=std

z = RBF(1.0).__call__(X)[0]

# Plot
ax.scatter(X[:, 0], X[:, 1], z, c=y, s=30, cmap=plt.cm.Paired)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.title('RBF Kernel')
# plt.savefig('./img/RBFkernel.png')
# plt.show()
plt.clf()

#---------------------------------------------------------------------------------------
# 3. Train SVM
#---------------------------------------------------------------------------------------

def plot_svc_decision_function(model,ax=None):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    Y,X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    ax.contour(X, Y, P,colors="k",levels=[-1,0,1],alpha=0.5,linestyles=["--","-","--"])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

X,y=make_circles(factor=0.2,noise=0.1) #factor = R2/R1, noise=std
clf = SVC(kernel="rbf").fit(X,y)
plt.scatter(X[:,0],X[:,1],c=y,s=30,cmap=plt.cm.Paired)
plot_svc_decision_function(clf)
plt.title('Train SVM')
plt.savefig('./img/RBFkernelSVM.png')
plt.show()                        
