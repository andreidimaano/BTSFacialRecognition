import face_recognition
import cv2

input_movie = cv2.VideoCapture("./assets/iu.mp4")
# length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
length = 720

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output_movie = cv2.VideoWriter('output.MP4', 0x7634706d, 24.0, (640, 360))

iu_image = face_recognition.load_image_file("./assets/iu.jpeg")
iu_face_encoding = face_recognition.face_encodings(iu_image)[0]

jk_image = face_recognition.load_image_file("./assets/jk.jpeg")
jk_face_encoding = face_recognition.face_encodings(jk_image)[0]

known_faces = [
    iu_face_encoding,
    jk_face_encoding
]

face_locations = []
face_encodings = []
face_names = []
frame_number = 0

# while True:
for i in range(0, length):
    ret, frame = input_movie.read()
    frame_number += 1

    if not ret:
        break

    rgb_frame = frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    valid_names = ["IU", "RM", "Jin", "Suga", "J-Hope", "Jimin", "V", "Jungkook"]
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = None
        if match[0]:
            name = "IU"
        elif match[1]:
            name = "Jungkook"

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        if name not in valid_names:
            cv2.rectangle(frame, (left, top), (right, bottom), (141, 153, 174), 2)
        else :
            cv2.rectangle(frame, (left, top), (right, bottom), (128, 237, 153), 2)
        
        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        if (name == "IU"): 
            cv2.putText(frame, "IU", (right + 5, top+5), font, 0.3, (128, 237, 153), 1)
        else: 
            cv2.putText(frame, "IU", (right + 5, top+5), font, 0.3, (255, 255, 255), 1)
        
        if (name == "RM"):
            cv2.putText(frame, "RM", (right + 5, top+15), font, 0.3, (128, 237, 153), 1)
        else:
            cv2.putText(frame, "RM", (right + 5, top+15), font, 0.3, (255, 255, 255), 1)

        if (name == "Jin"):
            cv2.putText(frame, "Jin", (right + 5, top+25), font, 0.3, (128, 237, 153), 1)
        else:
            cv2.putText(frame, "Jin", (right + 5, top+25), font, 0.3, (255, 255, 255), 1)
        
        if (name == "Suga"):
            cv2.putText(frame, "Suga", (right + 5, top+35), font, 0.3, (128, 237, 153), 1)
        else:
            cv2.putText(frame, "Suga", (right + 5, top+35), font, 0.3, (255, 255, 255), 1)
        
        if (name == "J-Hope"):
            cv2.putText(frame, "J-Hope", (right + 5, top+45), font, 0.3, (128, 237, 153), 1)
        else:
            cv2.putText(frame, "J-Hope", (right + 5, top+45), font, 0.3, (255, 255, 255), 1)

        if (name == "Jimin"):
            cv2.putText(frame, "Jimin", (right + 5, top+55), font, 0.3, (128, 237, 153), 1)
        else:
            cv2.putText(frame, "Jimin", (right + 5, top+55), font, 0.3, (255, 255, 255), 1)

        if (name == "V"):
            cv2.putText(frame, "V", (right + 5, top+65), font, 0.3, (128, 237, 153), 1)
        else:
            cv2.putText(frame, "V", (right + 5, top+65), font, 0.3, (255, 255, 255), 1)
            
        if (name == "Jungkook"):
            cv2.putText(frame, "Jungkook", (right + 5, top+75), font, 0.3, (128, 237, 153), 1)
        else:
            cv2.putText(frame, "Jungkook", (right + 5, top+75), font, 0.3, (255, 255, 255), 1)
            
        

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

input_movie.release()
cv2.destroyAllWindows()