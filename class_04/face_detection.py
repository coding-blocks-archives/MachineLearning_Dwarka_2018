




import cv2

im = cv2.imread('iu.jpeg', cv2.IMREAD_COLOR)
print(im.shape)

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
	ret, frame = cap.read() # Status, Frame

	if not ret:
		continue

	cv2.imshow("Feed", frame)

	key = cv2.waitKey(1)
	if key & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
