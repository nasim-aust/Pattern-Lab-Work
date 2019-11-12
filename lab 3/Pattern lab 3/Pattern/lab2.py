import numpy as np

"""
lambda_test= lambda a : a+10
print(lambda_test(5))
new_lambda_test=lambda q,w,y,z:q+w+y+z+10
print(new_lambda_test(20,20,20,20))

pairs=[(1,'one'),(2,'two'),(3,'three'),(4,'four')]
pairs.sort(key=lambda pair: pair[1])
print(pairs)

def make_incrementor(n):
    return lambda x:x+n
f=make_incrementor(42)
print(f(1))



list1=[1,2,3,4,5,6]
list2=[10,9,8,7,6,5]
a1=np.array(list1)
a2=np.array(list2)
print(a1*a2)

a=np.array([1,2,3,4,5], dtype=np.float64)
print(a,a.dtype,a.ndim,a.shape)
a=np.zeros((3,3))
print(a)
a=np.empty((4,4))
print(a)


print(np.linspace(0,10,5))
print(np.arange(0,10,2))

b=np.arange(12).reshape(4,3)
print(b)

c=np.arange(24).reshape(2,3,4)
print(c)
b=np.arange(12).reshape(3,4)
print(b)

"""
print(b.sum(axis=0))
print(b.min(axis=1))

"""
a=np.arange(10)**3
print(a)
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
print(a[:,0])
print(a[:,1])
print(a[:,2])

a=np.array(20)
print(a)

a=np.array([[1,2],[3,4],[5,6]])"""
