import os
import sys
import math
import pickle
import numpy as np
import cv2
import tensorflow as tf
import dlib
import collections
import argparse
from sklearn.svm import SVC
from tensorflow.python.platform import gfile
from tensorflow.python.training import training

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y  

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

def load_model(model, input_map=None):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with gfile.FastGFile(model_exp,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, input_map=input_map, name='')
    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)
        
        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
      
        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help = 'Path of the image you want to test on.',required=True)
    args = parser.parse_args()
	
    CLASSIFIER_PATH = './models/classifier.pkl'
    VIDEO_PATH = ''
    FACE_MODE_PATH = './models/20180402-114759.pb'
    predictor_path = './shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    file_name = args.path

    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            load_model(FACE_MODE_PATH)
            print('Loading feature extraction model')

            #get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            people_detected = set()
            person_detected = collections.Counter()
            image = cv2.imread(file_name);

            face_found = detector(image,1)
            faces = dlib.full_object_detections()

            for detection in face_found:
                faces.append(sp(image, detection))

            for k, detection in enumerate(face_found):
                # get toa do cua face
                (x, y, w, h) = rect_to_bb(detection)
                #lay phan face da can chinh lai
                aligned_face = dlib.get_face_chip(image, faces[k], size=160)
                aligned_face = prewhiten(aligned_face) # normalizes the range of the pixel values of input images.
                cv2.imshow("prewhite", aligned_face)
                scaled_reshape = aligned_face.reshape(-1, 160, 160, 3)
                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                best_name = class_names[best_class_indices[0]]
                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0),2)
                cv2.putText(image, "{} | {}".format(best_name, str(round(best_class_probabilities[0], 3))),
                 (x -10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)

            cv2.imshow("face", image)
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        
main()
