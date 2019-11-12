import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from matplotlib import style
style.use("ggplot")
"""x=np.linspace(0,20,200)
y1=np.exp(-0.1*x)*np.sin(x)
y2=np.exp(-0.3*x)*np.sin(x)
plt.plot(x,y1)
plt.plot(x,y2)
plt.title('Just enough!')
plt.show()"""
x=np.array([[1,2],
            [5,8],
            [1.5,1.8],
            [8,8],
            [1,0.6],
            [9,11]])
y=[0,1,0,1,0,1]
clf=svm.SVC(kernel='linear',C=1.0)
clf.fit(x,y)
#print(clf.predict(np.array([10,10]).reshape(1,-1)))
w=clf.coef_[0]
a=-w[0]/w[1]
xx=np.linspace(0,12).reshape(-1,1)
yy=a*xx -(clf.intercept_[0])/w[1]
plt.plot(xx,yy,'k--',label='Decision Boundary')
plt.scatter(x[:,0],x[:,1])
plt.legend()
plt.show()


