#!/usr/bin/python

""" Complete the code in ClassifyNB.py with the sklearn
    Naive Bayes classifier to ClassifyNB the terrain data.
    
    The objective of this exercise is to recreate the decision 
    boundary found in the lesson video, and make a plot that
    visually shows the decision boundary """

from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture, output_image
from ClassifyNB import classify
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()

# You will need to complete this function imported from the ClassifyNB script.
# Be sure to change to that code tab to complete this quiz.
clf = classify(features_train, labels_train)
pred = clf.predict(features_test)
accu = accuracy_score(labels_test, pred)
print "accuracy {}".format(accu)

# draw the decision boundary with the text points overlaid
prettyPicture(clf, features_test, pred)

output_image("test.png", "png", open("test.png", "rb").read())
