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

def main():
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    VIDEO_PATH = 'two.mp4'
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    cap = cv2.VideoCapture(VIDEO_PATH)

    while True:
        ret, frame = cap.read()
        faces = detector(frame, 1)
        for k, d in enumerate(faces):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
    #get landmarks for the face in box d.
            shape = sp(frame, d)
            shape = shape_to_np(shape)

            (x, y, w, h) = rect_to_bb(d)
            cv2.rectangle(frame, (x, y), (x + w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face #{}".format(k +1), (x -10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)

            for(x,y) in shape:
                cv2.circle(frame, (x,y), 1,(0,0,255), -1)

        cv2.imshow("Face Detector", frame)
        if cv2.waitKey(0):
            break

    cap.release()
    cv2.destroyAllWindows()

main()