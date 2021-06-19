import csv
import math
import os

import face_recognition
import numpy
from sklearn import svm

encodings = []
names = []

with open("encodings.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        encodings.append(row)

with open("names.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        names.append(row[0])

clf = svm.SVC(gamma='scale', probability=True)
clf.fit(encodings,names)

# Load the test image with unknown faces into a numpy array
test_image = face_recognition.load_image_file('./assets/bts.jpeg')

# Find all the faces in the test image using the default HOG-based model
face_locations = face_recognition.face_locations(test_image)
face_encodings = face_recognition.face_encodings(test_image, face_locations)

face_names = []
probabilites = []
valid_names = ["IU", "RM", "Jin", "Suga", "J-Hope", "Jimin", "V", "Jungkook"]

for face_encoding in face_encodings:
    name = clf.predict([face_encoding])
    face_names.append(*name)
    # print(*name)
    results = clf.predict_proba([face_encoding])[0]
    results = [str(int(element * 100)) for element in results]
    prob_per_class_dictionary = dict(zip(clf.classes_, results))
    probabilites.append(prob_per_class_dictionary)
    # print("probability pre class dictionary: ")
    # print(prob_per_class_dictionary)

for prob, name in zip(probabilites, face_names):
    print('\n')
    print("Prediction: " + name)
    for label in valid_names:
        if(len(prob[label]) == 1):
            prob[label] = "0" + prob[label]
        prob[label] = prob[label] +  "%:  " + label
    for i in range(len(valid_names)):
        print(prob[valid_names[i]])
