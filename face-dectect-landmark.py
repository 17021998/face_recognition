import dlib
import cv2
import numpy as np

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords

predictor_path = 'shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'dlib_face_recognition_resnet_model_v1.dat' # model
faces_folder_path = ''

#face detector
detector = dlib.get_frontal_face_detector()

#face analysis
sp = dlib.shape_predictor(predictor_path)

#face recognition
facerec = dlib.face_recognition_model_v1(face_rec_model_path)

image = cv2.imread('t1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#get all faces in picture
faces = detector(image, 1)
print("Number of faces detected: {}".format(len(faces)))

for k, d in enumerate(faces):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
    
    #get landmarks for the face in box d.
    shape = sp(gray, d)
    shape = shape_to_np(shape)

    (x, y, w, h) = rect_to_bb(d)
    cv2.rectangle(image, (x, y), (x + w, y+h), (0, 255, 0), 2)

    cv2.putText(image, "Face #{}".format(k +1), (x -10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)

    for(x,y) in shape:
        cv2.circle(image, (x,y), 1,(0,0,255), -1)

cv2.imshow("Output", image)

cv2.waitKey(0)




