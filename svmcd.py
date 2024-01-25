# Import the necessary modules
from sklearn import svm
from sklearn.impute import KNNImputer
import numpy as np
import mlcdconfig

# Read the data from a csv file
D = np.genfromtxt('data.csv', delimiter=',')
# X is the array without the last column for whichn the last column is not empty
X=D[:mlcdconfig.rows1-1,:mlcdconfig.columns-1]
print("X=",X)
# y is the last column of the array when it is not empty
y=D[:mlcdconfig.rows1-1,mlcdconfig.columns-1]
print("y",y)

# X_new is the array with the missing labels
X_new=D[mlcdconfig.rows1-1:mlcdconfig.rows1+mlcdconfig.rows2-1,:mlcdconfig.columns-1]
print("X_new=",X_new)
                                    
# Use the KNNImputer to impute the missing values
imputer = KNNImputer(n_neighbors=2)
X_imputed = imputer.fit_transform(X)
print("X_imputed=",X_imputed)

# Use the SVC with the RBF kernel to fit and predict the data
svc = svm.SVC(kernel='rbf', C=1, gamma=1)

## Use the SVC with the RBF kernel to fit and predict the data
svc.fit(X_imputed, y)                
y_pred = svc.predict(X_imputed)

# How good is the prediction
print(np.mean(y_pred == y))
# what is the accuracy of the prediction
print(svc.score(X_imputed, y))

# Print the predictions
print(y_pred)

X_new_imputed = imputer.fit_transform(X_new)

# Use the SVC with the RBF kernel to predict the data
               
y_new_pred = svc.predict(X_new_imputed)

# Print the predictions
print(y_new_pred)
