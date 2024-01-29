# Import the necessary modules
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
import mlcdconfig

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

    # Create a TfidfVectorizer to transform the text into vectors
    # As the default only delivers an accuracy of 50%, we need to optimize parameters for the vectorizer
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words=stop_words)),
        ('clf', OneVsRestClassifier(MultinomialNB(
            fit_prior=True, class_prior=None))),
    ])

    parameters = {
        'tfidf__max_df': (0.2, 0.25, 0.5, 0.75),
        'tfidf__ngram_range': [(1,1), (1,2), (1,3), (1,4)],
        'tfidf__use_idf': (True, False),
        'clf__estimator__alpha': (2e-2, 1e-2, 1e-3)
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

    # for each of the wrong predictions print the articletext, the correct category and the predicted category
    for i in range(0,mlcdconfig.rows2):
        if test_categories[i] != test_predictions[i]:
            print(test_articletexts[i] +"|"+ test_categories[i] +"|"+ test_predictions[i]+"|"+str(vectorizer.predict_proba([test_articletexts[i]]).max()))
            # print also the probability of the predicted category