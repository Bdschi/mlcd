# Import the necessary modules
from sklearn import svm
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
import numpy as np
import mlcdconfig

# Read the data from a csv file
D = np.genfromtxt('data.csv', delimiter=',')
# X is the array without the last column for whichn the last column is not empty
X=D[:mlcdconfig.rows1-1,:mlcdconfig.columns-1]
#print("X=",X)
# y is the last column of the array when it is not empty
y=D[:mlcdconfig.rows1-1,mlcdconfig.columns-1]
#print("y",y)

# X_new is the array with the missing labels
X_new=D[mlcdconfig.rows1-1:mlcdconfig.rows1+mlcdconfig.rows2-1,:mlcdconfig.columns-1]
#print("X_new=",X_new)
                                    
# Use the KNNImputer to impute the missing values
imputer = KNNImputer(n_neighbors=2)
X_imputed = imputer.fit_transform(X)
#print("X_imputed=",X_imputed)

# use  GridSearchCV to find the best parameters
parameters = {'C': [0.5, 1, 2, 5, 10, 20, 50, 100], 
           'gamma': [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]}
model = svm.SVC()
#grid = GridSearchCV(estimator=model, param_grid=parameters, n_splits=2, cv=5, verbose=1)
grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5, verbose=1)
grid.fit(X, y)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_estimator_)

## Use the SVC with the RBF kernel to fit and predict the data
y_pred = grid.predict(X_imputed)

# what is the accuracy of the prediction
print("score:",grid.score(X_imputed, y))

# Print the predictions
#print(y_pred)

X_new_imputed = imputer.fit_transform(X_new)

# Use the SVC with the RBF kernel to predict the data
               
y_new_pred = grid.predict(X_new_imputed)

# Print the predictions
#print(y_new_pred)
