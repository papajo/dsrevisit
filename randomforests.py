#%%
from sklearn.ensemble import RandomForestClassifier #use RandomForestRegressor for regression problem
#Assume you have X(Predictor) and Y(target) for training dataset and x_test(predictor) of test_dataset
#Create RandomForest object
model = RandomForestClassifier(n_estimators=1000)
#Train the model using the training set and check score
model.fit(X, y)
#Predict output
predicted = model.predict(x_test)