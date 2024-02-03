"""Package for training detectron2 model for detecting weightlifting plates.

Package also includes detection of lifter, plates, 
and outputs video overlayed with detection and calculates speed of bar during lift."""

# Importing modules
from .custom_object_trainer import Custom_Object_Trainer
from .local_detector import Detector
from .json_to_d2 import json_to_d2
from .aws_detector import AWSDetector