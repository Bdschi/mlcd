# Import the necessary modules
import gensim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the pre-trained word2vec model from Google
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Define the text and the categories
text = ["This is a positive sentence",
"This is a negative sentence",
"This is a neutral sentence"]
categories = [1, 0, 0]

# Define a function to convert text to vectors using word2vec
def text_to_vectors(text, model):
    # Initialize an empty list to store the vectors
    vectors = []
    # Loop through the text
    for sentence in text:
        # Split the sentence into words
        words = sentence.split()
        # Initialize an empty array to store the word vectors
        word_vectors = []
        # Loop through the words
        for word in words:
            # Check if the word is in the model's vocabulary
            if word in model:
                # Get the word vector and append it to the word vectors array
                word_vector = model[word]
                word_vectors.append(word_vector)
        # Average the word vectors to get the sentence vector
        sentence_vector = np.mean(word_vectors, axis=0)
        # Append the sentence vector to the vectors list
        vectors.append(sentence_vector)
    # Return the vectors list
    return vectors

# Convert the text to vectors using the word2vec model
vectors = text_to_vectors(text, model)

# Split the data into training and test sets
X_train = vectors[:2]
X_test = vectors[2:]
y_train = categories[:2]
y_test = categories[2:]

# Create a logistic regression classifier
classifier = LogisticRegression()
# Fit the classifier on the training data
classifier.fit(X_train, y_train)
# Predict the category for the test data
y_pred = classifier.predict(X_test)

# Print the prediction and the accuracy score
print("Prediction:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
