import cv2
import numpy as np
import time

name = input("Enter Name: ")
num = int(input("Number of Photos: "))
face_data = []

cap = cv2.VideoCapture(0)

# Instantiate the Cascade Classifier with file_name
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True and num:
	time.sleep(0.5) # Sleep for 0.5 seconds
	ret, frame = cap.read() # Status, Frame

	if not ret:
		continue


	cv2.imshow("Feed", frame)
	# Find all the faces in the frame
	faces = face_cascade.detectMultiScale(frame, 1.3, 5) # Frame, Scaling Factor, Neighbors
	
	# Taking only the face of the person (assumption: greater area)
	faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
	faces = faces[:1]

#	print(faces)

	for face in faces:
		x,y,w,h = face # Tuple Unpacking

		# Drawing Boundary
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2) # Frame, Start, End, Color,Thickness

		face_only = frame[y:y+h, x:x+w]
		face_only = cv2.resize(face_only, (100,100))
		face_data.append(face_only)
		num -= 1
		
#		cv2.imshow("Face Selection", face_only)


	key = cv2.waitKey(1)
	if key & 0xFF == ord('q'):
		break

print(len(face_data))
face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)
np.save( ("face_dataset/" + name), face_data)

cap.release()
cv2.destroyAllWindows()
