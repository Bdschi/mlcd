import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Load the data
data = np.loadtxt('articles.txt', delimiter='\t', dtype='float32') # Assuming the data is stored in 'articles.txt'
X = data[:, :-1] # Features are stored in the first 'n-1' columns
y = data[:, -1] # Category is stored in the last column

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create the decision tree classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Load an article to test
article_features = [10, 20, 30] # Replace with the actual features of the article

# Predict the category of the article
predicted_category = classifier.predict([article_features])[0]

print('The predicted category of the article is:', predicted_category)

