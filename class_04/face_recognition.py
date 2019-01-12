import os
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

### COLLECTING DATA ###

files = [f for f in os.listdir('face_dataset') if f.endswith('.npy')]
names = [f[:-4] for f in files]

face_data = []

for filename in files:
	data = np.load('face_dataset/'+filename)
#	print(data.shape)
	face_data.append(data)

face_data = np.concatenate(face_data, axis=0)
print(face_data.shape)
print(names)
names = np.array(names)
names = np.repeat(names, 10)
names = names.reshape((-1,1))


dataset = np.hstack((face_data, names))


### Training Classifier ###
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(dataset[:, :-1], dataset[:, -1])


# Testing
cap = cv2.VideoCapture(0)

# Instantiate the Cascade Classifier with file_name
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
	ret, frame = cap.read() # Status, Frame

	if not ret:
		continue
	# Find all the faces in the frame
	faces = face_cascade.detectMultiScale(frame, 1.3, 5) # Frame, Scaling Factor, Neighbors

#	print(faces)

	for face in faces:
		x,y,w,h = face # Tuple Unpacking

		face_only = frame[y:y+h, x:x+w]
		face_only = cv2.resize(face_only, (100,100))
		face_only = face_only.reshape((1,-1))
		
		print(face_only.shape)
		pred = knn.predict(face_only)
		print(pred)
		
		# Drawing Boundary
		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2) # Frame, Start, End, Color,Thickness
		cv2.putText(frame, pred[0], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
		
	cv2.imshow("Feed", frame)

	key = cv2.waitKey(1)
	if key & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
