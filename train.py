import face_recognition
import numpy
import os
import csv

# Training the SVC classifier

# The training data would be all the face encodings from all the known images and the labels are their names
encodings = []
names = []

# Training directory
train_dir = os.listdir('./train_dir/')

# Loop through each person in the training directory
for i in range(1, len(train_dir), 1):
    pix = os.listdir("./train_dir/" + train_dir[i])

    # Loop through each training image for the current train_dir[i]
    # for j in range(0, len(pix), 30):
    j = 0
    for img in pix:
        if j > 300:
            break
        # Get the face encodings for the face in each image file
        face = face_recognition.load_image_file("./train_dir/" + train_dir[i] + "/" + img)
        # face = face_recognition.load_image_file("./train_dir/" + train_dir[i] + "/" + pix[j])
        face_bounding_boxes = face_recognition.face_locations(face)

        #If training image containsv exactly one face
        if len(face_bounding_boxes) == 1:
            print(img + "...")
            face_enc = face_recognition.face_encodings(face)[0]
            # Add face encoding for current image with corresponding label (name) to the training data
            encodings.append(face_enc)
            names.append(train_dir[i])
        else:
            print(train_dir[i] + "/" + img + " was skipped and can't be used for training")
            # print(train_dir[i] + "/" + pix[j]+ " was skipped and can't be used for training")
        j += 1

# Create and train the SVC classifier
# print(encodings)
# print(names)
numpy.savetxt("encodings.csv", encodings, delimiter = ",")
with open('names.csv', 'w') as result_file:
    wr = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for name in names:
        wr.writerow([name])