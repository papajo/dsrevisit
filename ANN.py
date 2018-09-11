#%%
# Artificial Neural Networks in Machine Learning
# Class MLP Classifier implements a multi-layer perceptron (MLP) algorithm that trains
# using Back propagation(gradient descent)
from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2),random_state=1)
clf.fit(X, y)
clf.predict([[2., 2.], [-1., -2.]])
[coef.shape for coef in clf.coefs_]
clf.predict_proba([[2., 2.], [1., 2.]])
# MLP Classifier supports multi-class classification by applying Softmax as the output
# function.