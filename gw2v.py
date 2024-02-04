
import gensim
from gensim.models import Word2Vec
from sklearn.svm import SVC
import re

def clean_punc(sentence):
  cleaned=re.sub(r'[?|!|\'|"|#]',r'',sentence)
  cleaned=re.sub(r'[.|,)|(|\|/]',r' ',cleaned)
  return cleaned

# Load your text data (replace with your actual data loading)
texts = [
    "This is a text about sports. It discusses football, basketball, and baseball.",
    "This text is about politics. It talks about elections, governments, and policies.",
    "This is a text about technology. It covers computers, coding, and artificial intelligence.",
    # ... more texts
]
# List of corresponding category labels
labels = [
        "sport",
        "politics",
        "technology",
]
# Preprocess the texts (tokenize, remove stop words, etc.)
tokenized_texts = [clean_punc(text).split() for text in texts]
# Train the Word2Vec model
model = Word2Vec(tokenized_texts, min_count=1)
# Create document vectors by averaging word vectors
doc_vectors = []
for text in tokenized_texts:
  doc_vector = sum(model.wv[word] for word in text) / len(text)
  doc_vectors.append(doc_vector)
# Train the SVM classifier
clf = SVC()
clf.fit(doc_vectors, labels)
# Predict categories for new texts
new_texts = [
  "sports.",
]
new_doc_vectors = [sum(model.wv[word] for word in clean_punc(text).split()) / len(clean_punc(text).split()) for text in new_texts]
predicted_categories = clf.predict(new_doc_vectors)

print("Predicted categories for new texts:", predicted_categories)
