
import gensim
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import mlcdconfig
import re
import csv
import random

def clean_punc(sentence):
  cleaned=re.sub(r'[?|!|\'|"|#]',r'',sentence)
  cleaned=re.sub(r'[.|,)|(|\|/]',r' ',cleaned)
  return cleaned

level=4
articletexts = []
categories = []

file="/mnt/c/Users/bdschi/Downloads/artikelstruktur_(Artikeltext)_558552.csv"
with open(file, "r") as csvfile:
  lines = csv.reader(csvfile, delimiter=';', quotechar='"')
  for line in lines:
      text = line[0]
      if text[0] == '"':
          text=text[1:]
      if text[-1] == '"':
          text=text[:-1]
      atexts=text.split("\\")
      if len(atexts) < 6: 
          continue
      if atexts[level][0] == " ":
          atexts[level]=atexts[level][1:]
      if atexts[5][0] == " ": 
          atexts[5]=atexts[5][1:]
      #print("articletext=\"%s\" category=\"%s\"" % (atexts[5], atexts[level]))
      articletexts.append(atexts[5])
      categories.append(atexts[level])

  print("Data loaded")
  randomindices=[i for i in range(0, len(articletexts))]
  random.shuffle(randomindices)

  # Split the data into training and test sets
  # TypeError: list indices must be integers or slices, not list
  # create a list of all articletexts with indices in randomindices[:mlcdconfig.rows1]
  train_articletexts = [articletexts[i] for i in randomindices[:mlcdconfig.rows1]]
  train_categories = [categories[i] for i in randomindices[:mlcdconfig.rows1]]
  test_articletexts = [articletexts[i] for i in randomindices[mlcdconfig.rows1:mlcdconfig.rows1+mlcdconfig.rows2]]
  test_categories = [categories[i] for i in randomindices[mlcdconfig.rows1:mlcdconfig.rows1+mlcdconfig.rows2]]

  print("Data split")
  # Preprocess the texts (tokenize, remove stop words, etc.)
  tokenized_texts = [clean_punc(text).split() for text in articletexts]
  # Train the Word2Vec model
  model = Word2Vec(tokenized_texts, min_count=1)
  # Create document vectors by averaging word vectors
  print("word2vec trained")

  # Train the SVM classifier
  tokenized_train_texts = [clean_punc(text).split() for text in train_articletexts]
  doc_vectors = []
  for text in tokenized_train_texts:
      doc_vector = sum(model.wv[word] for word in text) / len(text)
      doc_vectors.append(doc_vector)
  clf = SVC()
  clf.fit(doc_vectors, train_categories)
  print("SVC trained")

  # Predict categories for new texts
  new_doc_vectors = [sum(model.wv[word] for word in clean_punc(text).split()) / len(clean_punc(text).split()) for text in test_articletexts]
  predicted_categories = clf.predict(new_doc_vectors)

  for i in range(0, len(test_categories)):
      if test_categories[i] != predicted_categories[i]:
          print("Text=\"%s\" Test category=\"%s\" predicted category=\"%s\"" % (test_articletexts[i], test_categories[i], predicted_categories[i]))
  
  print("Accuracy: %.2f" % accuracy_score(test_categories, predicted_categories))