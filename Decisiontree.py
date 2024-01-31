import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import mlcdconfig
import csv
import random

level=2
categories=[]
features=[]
pgs=[]
file="/mnt/c/Users/bdschi/Downloads/artikelstruktur_(pg_measures)_559510.csv"
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
        pgs.append(atexts[5])
        measures=[]
        for im in range(2, 5):
            measures.append(float(line[im].replace(",", ".")))
        #print("category=\"%s\" pg=\"%s\" measures=%s" % (atexts[level], atexts[5], measures))
        categories.append(atexts[level])
        features.append(measures)
    print("number of different product groups %d" % len(set(pgs)))
    pgset=set(pgs)
    #for i in range(len(pgset)):
    #    print("i=%d pg=\"%s\"" % (i, list(pgset)[i]))
    newfeatures=[]
    emptypgs=len(set(pgs))*[0.0]
    for i in range(len(categories)):
        newfeature=features[i]+emptypgs
        #print("pg=\"%s\" index=%d" % (pgs[i], list(pgset).index(pgs[i])))
        newfeature[3+list(pgset).index(pgs[i])]=1.0
        newfeatures.append(newfeature)
        #print("newfeature=%s" % newfeature)
    randomindices=[i for i in range(0, len(categories))]
    random.shuffle(randomindices)
    train_features = [newfeatures[i] for i in randomindices[:mlcdconfig.rows1]]
    train_categories = [categories[i] for i in randomindices[:mlcdconfig.rows1]]
    test_features = [newfeatures[i] for i in randomindices[mlcdconfig.rows1:mlcdconfig.rows1+mlcdconfig.rows2]]
    test_categories = [categories[i] for i in randomindices[mlcdconfig.rows1:mlcdconfig.rows1+mlcdconfig.rows2]]

    # Create the decision tree classifier
    classifier = DecisionTreeClassifier()
    classifier.fit(train_features, train_categories)

    # Make predictions on the test data
    predictions = classifier.predict(test_features)

    totcnt=0
    errcnt=0
    # Print out the actual and predicted categories
    for i in range(len(test_categories)):
        totcnt+=1
        if(test_categories[i] != predictions[i]):
            errcnt+=1
            print("actual=\"%s\" predicted=\"%s\"" % (test_categories[i], predictions[i]))

    print("total=%d errors=%d accuray=%.2f" % (totcnt, errcnt, (totcnt-errcnt)/totcnt))