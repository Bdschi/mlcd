# Import the necessary modules
from sklearn import svm
from sklearn.impute import KNNImputer
import numpy as np

# Read the data from a csv file
D = np.genfromtxt('data.csv', delimiter=',')
# X is the array without the last column for whichn the last column is not empty
X = D[-np.isnan(D).any(axis=1), :-1]
# y is the last column of the array when it is not empty
y = D[-np.isnan(D).any(axis=1), -1]

# X_new is the array with the missing labels
X_new = D[np.isnan(D).any(axis=1), :-1]
                                    
# Use the KNNImputer to impute the missing values
imputer = KNNImputer(n_neighbors=2)
X_imputed = imputer.fit_transform(X)

# Use the SVC with the RBF kernel to fit and predict the data
svc = svm.SVC(kernel='rbf')
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
