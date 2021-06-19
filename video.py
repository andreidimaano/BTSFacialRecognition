import face_recognition
import numpy
from sklearn import svm
import os
import cv2
import csv

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

with open("encodings2.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
    for row in reader:
        encodings.append(row)

with open("names2.csv") as csvfile:
    reader = csv.reader(csvfile, quoting=csv.QUOTE_MINIMAL)
    for row in reader:
        names.append(row[0])

clf = svm.SVC(gamma='scale', probability=True)
clf.fit(encodings,names)

# Load the test image with unknown faces into a numpy array
# test_image = face_recognition.load_image_file('./assets/bts.jpeg')
# load movie
input_movie = cv2.VideoCapture("./assets/HQDynamite.mp4")
height = int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH))
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
# length = 24 * 10

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output_movie = cv2.VideoWriter('test.mp4', 0x7634706d, 24.0, (width,height))
  
# Find all the faces in the test image using the default HOG-based model
# face_locations = face_recognition.face_locations(test_image)
# no = len(face_locations)
# print("Number of faces detected: ", no)

face_names = []
frame_number = 0
valid_names = ["IU", "RM", "Jin", "Suga", "J-Hope", "Jimin", "V", "Jungkook", "other"]

while True:
# for i in range(0, length):
    ret, frame = input_movie.read()
    frame_number += 1

    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    probabilites = []
    for face_encoding in face_encodings:
        # Predict the person
        name = clf.predict([face_encoding]) 
        
        results = clf.predict_proba([face_encoding])[0]
        results = [str(int(element * 100)) for element in results]
        prob_per_class_dictionary = dict(zip(clf.classes_, results))
        probabilites.append(prob_per_class_dictionary)
        
        probabilites.append(prob_per_class_dictionary)
        face_names.append(*name)

    # Draw Boxes on Video
    for (top, right, bottom, left), name, prob in zip(face_locations, face_names, probabilites):
        if not name:
            continue
        
        for label in valid_names:
            if(len(prob[label]) == 1):
                prob[label] = " " + prob[label]

        # Draw a box around the face
        if name == "other":
            cv2.rectangle(frame, (left, top), (right, bottom), (141, 153, 174), 2)
        else :
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 114, 0), 2)
        
        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX


        for i in range(1, len(valid_names)):
            name_drawing = prob[valid_names[i]] + "% " + valid_names[i]
            if(name == valid_names[i]):
                if name == "other":
                    cv2.putText(frame, name_drawing, (right + 5, (top+20) + (30 * (i - 1))), font, 0.75, (141, 153, 174), 1)
                else:    
                    cv2.putText(frame, name_drawing, (right + 5, (top+20) + (30 * (i - 1))), font, 0.75, (0, 114, 0), 1)
            else :
                cv2.putText(frame, name_drawing, (right + 5, top+20 + (30 * (i - 1))), font, 0.75, (255, 255, 255), 1)

     # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

input_movie.release()
cv2.destroyAllWindows()
