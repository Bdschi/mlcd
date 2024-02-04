import gensim
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

# Load your text data (replace with your actual data loading)
texts = [
    "This is a text about sports. It discusses football, basketball, and baseball.",
    "This text is about politics. It talks about elections, governments, and policies.",
    "This is a text about technology. It covers computers, coding, and artificial intelligence.",
    # ... more texts
]

# Preprocess the texts (tokenize, remove stop words, etc.)
tokenized_texts = [text.split() for text in texts]

# Train the Word2Vec model
model = Word2Vec(tokenized_texts, min_count=1)

# Create document vectors by averaging word vectors
doc_vectors = []
for text in tokenized_texts:
    doc_vector = sum(model.wv[word] for word in text) / len(text)
    doc_vectors.append(doc_vector)

# Categorize using K-Means clustering
num_clusters = 3  # Adjust based on your expected number of categories
kmeans = KMeans(n_clusters=num_clusters, n_init='auto', max_iter=100, random_state=42)
kmeans.fit(doc_vectors)

# Get predicted categories
predicted_categories = kmeans.labels_

print("Predicted categories:", predicted_categories)

