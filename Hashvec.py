# Import the necessary modules
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
import mlcdconfig
# from sklearn.model_selection import train_test_split

# Open the file and read the lines
with open("articletexts.txt", "r") as f:
    lines = f.readlines()

    # Split the lines into articletexts and categories
    articletexts = []
    categories = []
    for line in lines:
        sentence, category = line.split("\t")
        articletexts.append(sentence)
        categories.append(category.strip())

    # Split the data into training and test sets
    train_articletexts = articletexts[:mlcdconfig.rows1]
    train_categories = categories[:mlcdconfig.rows1]
    test_articletexts = articletexts[mlcdconfig.rows1:mlcdconfig.rows1+mlcdconfig.rows2]
    test_categories = categories[mlcdconfig.rows1:mlcdconfig.rows1+mlcdconfig.rows2]
    stop_words=['in', 'und']

    # Create a HashingVectorizer to transform the text into vectors
    pipeline = Pipeline([
        ('hashvec', HashingVectorizer(stop_words=stop_words)),
        ('clf', OneVsRestClassifier(MultinomialNB(
            fit_prior=True, class_prior=None))),
    ])
    parameters = {
        'hashvec__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'clf__estimator__alpha': (1e-2, 1e-3)
    }

    vectorizer = GridSearchCV(pipeline, parameters, cv=2, n_jobs=2, verbose=3)
    vectorizer.fit(train_articletexts, train_categories)

    print("Best parameters set:")
    print(vectorizer.best_estimator_.steps)
 
    # Transform the test articletexts using the same vectorizer
    # AttributeError: This 'Pipeline' has no attribute 'transform'
    # test_vectors = vectorizer.transform(test_articletexts)
  
    # Predict the categories for the test vectors
    test_predictions = vectorizer.predict(test_articletexts)

    # Print the accuracy score of the classifier
    print("Accuracy:", accuracy_score(test_categories, test_predictions))
