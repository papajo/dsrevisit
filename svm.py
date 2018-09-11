#%%
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

#Create 40 separable points
np.random.seed(0)
X = np.r_[np.random.randn(20,2) - [2,2], np.random.randn(20,2) + [2,2]]
Y = [0]*20 + [1]*20
#Fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)
#Get the separating hyperplane
w = clf.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0])/w[1]
#Plot the parallels to the separating hyperplane that pass thru the support vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])
#Plot the line the points and the nearest vectors to the plane
plt.plot(xx, yy, 'k-')
plt.plot(xx, yy_down, 'k--')
plt.plot(xx, yy_up, 'k--')
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
    s=80, facecolors='none')
plt.axis('tight')
plt.show()

