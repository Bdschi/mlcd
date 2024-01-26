# Import the necessary modules
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Open the file and read the lines
with open("sentences.txt", "r") as f:
    lines = f.readlines()

    # Split the lines into sentences and categories
    sentences = []
    categories = []
    for line in lines:
        sentence, category = line.split(",")
        sentences.append(sentence)
        categories.append(category.strip())

    # Split the data into training and test sets
    train_sentences = sentences[:80]
    train_categories = categories[:80]
    test_sentences = sentences[80:]
    test_categories = categories[80:]

    # Create a TfidfVectorizer to transform the text into vectors
    vectorizer = TfidfVectorizer()
    # Fit the vectorizer on the training sentences and transform them
    train_vectors = vectorizer.fit_transform(train_sentences)
    # Transform the test sentences using the same vectorizer
    test_vectors = vectorizer.transform(test_sentences)

    # Create a MultinomialNB classifier to predict the categories
    classifier = MultinomialNB()
    # Fit the classifier on the training vectors and categories
    classifier.fit(train_vectors, train_categories)
    # Predict the categories for the test vectors
    test_predictions = classifier.predict(test_vectors)

    # Print the accuracy score of the classifier
    print("Accuracy:", accuracy_score(test_categories, test_predictions))

    # q: can you correct intendation in the code?
    # a: yes, I can

