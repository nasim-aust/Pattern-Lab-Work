import numpy as np
from matplotlib import pyplot as plt
x=np.linspace(0,20,200)
y1=np.exp(-0.1*x)*np.sin(x)
y2=np.exp(-0.3*x)*np.sin(x)
plt.plot(x,y1)
plt.plot(x,y2)
plt.title('Just Enough!')
plt.show()