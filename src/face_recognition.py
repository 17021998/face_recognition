import os
import sys
import math
import pickle
import numpy as np
import cv2
import tensorflow as tf
import dlib
from sklearn.svm import SVC
from tensorflow.python.platform import gfile
from tensorflow.python.training import training

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
    CLASSIFIER_PATH = '../data/Friends/clasifier.pkl'
    VIDEO_PATH = ''
    FACE_MODE_PATH = './models/20180402-114759.pb'
    predictor_path = 'shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            load_model(FACE_MODE_PATH)

            #get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            people_detected = set()
            person_detected = collections.Counter()

            #READ VIDEO
            cap = cv2.VideoCapture(VIDEO_PATH)
            while(cap.isOpened()):
                ret, frame = cap.read()
                
                faces = detector(frame,1)