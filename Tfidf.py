# Import the necessary modules
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

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
    train_articletexts = articletexts[:80]
    train_categories = categories[:80]
    test_articletexts = articletexts[80:]
    test_categories = categories[80:]

    # Create a TfidfVectorizer to transform the text into vectors
    # As the default only delivers an accuracy of 50%, we need to parameters to the vectorizer
    # min_df: minimum number of documents a word must be present in to be kept
    # max_df: maximum number of documents a word can be present in to be kept
    # stop_words: remove the most common words in the English language
    # We set min_df to 1 and max_df to 50
    vectorizer = TfidfVectorizer(min_df=1, max_df=50, stop_words="english")
    # Fit the vectorizer on the training articletexts and transform them
    train_vectors = vectorizer.fit_transform(train_articletexts)
    # Transform the test articletexts using the same vectorizer
    test_vectors = vectorizer.transform(test_articletexts)

    # Create a MultinomialNB classifier to predict the categories
    classifier = MultinomialNB()
    # Fit the classifier on the training vectors and categories
    classifier.fit(train_vectors, train_categories)
    # Predict the categories for the test vectors
    test_predictions = classifier.predict(test_vectors)

    # Print the accuracy score of the classifier
    print("Accuracy:", accuracy_score(test_categories, test_predictions))
