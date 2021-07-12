import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X = np.load("X_miniimagenet5_50_ORB.npy")

X_embedded = TSNE(n_components=2).fit_transform(X)
label = np.squeeze(np.arange(5).reshape(1,5).repeat(50,1), axis=0)

plt.scatter(X_embedded[:,0],X_embedded[:,1] , c=label)
plt.show()
