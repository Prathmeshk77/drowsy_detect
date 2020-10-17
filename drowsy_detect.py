from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
#import vlc
from pygame import mixer

region = ["South", "North", "East", "West"]

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear

def detect_drowsi(thresh):
	frame_check = 10
	detect = dlib.get_frontal_face_detector()
	predict = dlib.shape_predictor(r'C:\Users\prath\Desktop\be_project\Drowsydetect\shape_predictor.dat') 

	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
	cap=cv2.VideoCapture(0)
	flag=0
	while True:
		ret, frame=cap.read()
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		subjects = detect(gray, 0)
		for subject in subjects:
			shape = predict(gray, subject)
			shape = face_utils.shape_to_np(shape)#converting to NumPy Array
			leftEye = shape[lStart:lEnd]
			rightEye = shape[rStart:rEnd]
			leftEAR = eye_aspect_ratio(leftEye)
			rightEAR = eye_aspect_ratio(rightEye)
			ear = (leftEAR + rightEAR) / 2.0
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
			cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
			print("EAR : " ,ear)
			if ear < thresh:
			
				if flag >= frame_check:
					cv2.putText(frame, "****************ALERT!****************", (10, 30),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					cv2.putText(frame, "****************ALERT!****************", (10,325),
						cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
					print ("Drowsy")
					# p = vlc.MediaPlayer(r'C:\Users\prath\Desktop\be_project\Drowsydetect/beep-02.mp3')
					# p.play()	
					mixer.init()
					mixer.music.load(r'C:\Users\prath\Desktop\be_project\Drowsydetect\alert.mp3')
					mixer.music.play()
				flag += 1
				print (flag)
			else:
				flag = 0
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

	cv2.destroyAllWindows()
	cap.stop()


while(1):
	age = int(input("Enter your Age: "))

	if age < 18:
		print("Please get an Adult to drive")
		break

	else:
		print("Good! Carry your Driving Lisence with you")
		
	print("Enter your Region")
	reg = input(" South, North, East, West: ")
	thresh = 0.3
	detect_drowsi(thresh)
	


