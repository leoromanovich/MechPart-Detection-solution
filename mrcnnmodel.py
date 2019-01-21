import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
import skimage


# Import Mask RCNN
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
import tensorflow as tf
# Root directory of the project
ROOT_DIR = os.path.abspath("")

sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
def load_model():
    class InferenceConfig(Config):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        NAME = "mech"
        GPU_COUNT = 1
        NUM_CLASSES = 1 + 1  # Background + detail
        IMAGES_PER_GPU = 1
        DETECTION_MIN_CONFIDENCE = 0.0
        BACKBONE = "resnet50"
        RPN_ANCHOR_STRIDE = 2
        POST_NMS_ROIS_INFERENCE = 500


    config = InferenceConfig()
    config.display()


    path_to_model = 'mrcnn.h5'
    with tf.device("/gpu:0"):
        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(path_to_model, by_name=True)

    return model


