#%% Use code to install dependencies
# Start venv in directory
# Install cuda toolkit 12.2 https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11
# Install in conda venv:
# make sure torch completely uninstalled to make sure cuda is installed
#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# If get microsoft visual c required, run pip
#%% Testing pytorch
import torch
print(f'PyTorch version: {torch.__version__}')
print("Is CUDA enabled?",torch.cuda.is_available())
print('*'*10)
#!nvcc --version
print('*'*10)
print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')
#%%
from detectron2.engine import DefaultPredictor
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.structures import Boxes
from detectron2 import model_zoo
# from PIL import Image 
# import PIL 
import cv2
import numpy as np
import tqdm
import random
import matplotlib.pyplot as plt
import os
import math
# %matplotlib inline
# %%
videoPath = 'Li Dayin 180kg Snatch WR  216kg Clean&Jerk  2023 AWC in Jinju.mp4'
#%%
# Downloading custom barbell dataset
#!pip install roboflow --user
# from roboflow import Roboflow
# rf = Roboflow(api_key="cBvYNRp5CDciSK1U0OdT")
# project = rf.workspace("computer-vision-ks9fx").project("barbell-small")
# dataset = project.version(2).download("coco")
#%%
# need to register custom data so it can be used for training
from detectron2.data.datasets import register_coco_instances
# Removing registered dataset in case a mistake is made
# DatasetCatalog.remove('barbell_train')
# DatasetCatalog.remove('barbell_val')
# DatasetCatalog.remove('barbell_test')
# format is register_coco_instances("YourTrainDatasetName", {},"path to train.json", "path to train image folder")
# for train, valid, and test
register_coco_instances("barbell_train",{},'Barbell-small-2/train/_annotations.coco.json', 'Barbell-small-2/train' )
register_coco_instances("barbell_val",{},'Barbell-small-2/valid/_annotations.coco.json', 'Barbell-small-2/valid' )
register_coco_instances("barbell_test",{},'Barbell-small-2/test/_annotations.coco.json', 'Barbell-small-2/test' )
#%%
# Custom training configs
cfg_obj = get_cfg() # obtain detectron2's default config
cfg_obj.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")) # load values from a file specified under Detectron2's official configs/ directory
cfg_obj.DATASETS.TRAIN = ("barbell_train",)
cfg_obj.DATASETS.TEST = ("barbell_val",)

cfg_obj.DATALOADER.NUM_WORKERS = 0 # originally 4, set to 0 to disable parallel loads
cfg_obj.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml") # returns the URL to the model trained using the given config
cfg_obj.SOLVER.IMS_PER_BATCH = 4
cfg_obj.SOLVER.BASE_LR = 0.001 # learning rate

cfg_obj.SOLVER.WARMUP_ITERS = 1000
cfg_obj.SOLVER.MAX_ITER = 1500 #adjust up if val mAP is still rising, adjust down if overfit
# Solver steps changed end of range from 1500 to 1400 because it has to be lower than max iter i think
cfg_obj.SOLVER.STEPS = (1000,1400)
cfg_obj.SOLVER.GAMMA = 0.05

cfg_obj.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
cfg_obj.MODEL.ROI_HEADS.NUM_CLASSES = 2 # adjust for number of classes

cfg_obj.TEST.EVAL_PERIOD = 500
#%%
run_train=False
if run_train:
  # Visualize training data
  barbell_train_metadata = MetadataCatalog.get("barbell_train")
  dataset_dicts = DatasetCatalog.get("barbell_train")

  for d in random.sample(dataset_dicts, 3):
      img = cv2.imread(d["file_name"])
      visualizer = Visualizer(img[:, :, ::-1], metadata=barbell_train_metadata, scale=0.5)
      vis = visualizer.draw_dataset_dict(d)
      cv2.imshow('barbell',vis.get_image()[:, :, ::-1])
      cv2.waitKey(0)
      cv2.destroyAllWindows()
if run_train:
  # Need to make sure out model validates against our validation set
  from detectron2.engine import DefaultTrainer
  from detectron2.evaluation import COCOEvaluator

  class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):

      if output_folder is None:
          os.makedirs("coco_eval", exist_ok=True)
          output_folder = "coco_eval"

      return COCOEvaluator(dataset_name, cfg, False, output_folder)
  os.makedirs(cfg_obj.OUTPUT_DIR, exist_ok=True)
  trainer = CocoTrainer(cfg_obj)
  trainer.resume_or_load(resume=False)
  trainer.train()
#%%
# Look at training curves in tensorboard:
# %load_ext tensorboard
# %tensorboard --logdir output
# Loss appeared to bottom out at 1400
#%%
if run_train:
  from detectron2.evaluation import COCOEvaluator, inference_on_dataset

  #cfg_obj.MODEL.WEIGHTS = os.path.join(cfg_obj.OUTPUT_DIR, "model_final.pth")
  cfg_obj.MODEL.WEIGHTS = 'output/model_final.pth'
  cfg_obj.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
  predictor = DefaultPredictor(cfg_obj)
  evaluator = COCOEvaluator("barbell_test", cfg_obj, False, output_dir="./output/")
  val_loader = build_detection_test_loader(cfg_obj, "barbell_test")
  inference_on_dataset(trainer.model, val_loader, evaluator)
#%%
# run on real images
#cfg_obj.MODEL.WEIGHTS = os.path.join(cfg_obj.OUTPUT_DIR, "model_final.pth")
cfg_obj.MODEL.WEIGHTS ='output/model_final.pth'
cfg_obj.DATASETS.TEST = ("barbell_test", )
cfg_obj.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# predictor = DefaultPredictor(cfg_obj)
barbell_metadata = MetadataCatalog.get("barbell_test")
# I believe box for object is top left corner, bottom right corner
# from detectron2.utils.visualizer import ColorMode
# import glob
# this code is for showing all images in test folder with predictions drawn over it
# for imageName in glob.glob('/content/detectron2/test/*jpg'):
#   im = cv2.imread(imageName)
#   outputs = predictor(im)
#   v = Visualizer(im[:, :, ::-1],
#                 metadata=test_metadata,
#                 scale=0.8
#                  )
#   out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#   cv2_imshow(out.get_image()[:, :, ::-1])
# obtain frame
# video = cv2.VideoCapture(videoPath)
# video.set(cv2.CAP_PROP_POS_FRAMES, 532) #0-based index of the frame to be decoded/captured next.
# ret, frame = video.read()
# predictor_obj = DefaultPredictor(cfg_obj)
# outputs_obj = predictor_obj(frame)
# outputs_obj
# %%
cfg_key = get_cfg() # obtain detectron2's default config
# may need to download config file locally for greater speed
cfg_key.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")) # load values from a file specified under Detectron2's official configs/ directory
cfg_key.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml") # returns the URL to the model trained using the given config
cfg_key.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # threshold to filter out low-scored bounding boxes
cfg_key.MODEL.DEVICE = 'cuda'  # cpu or cuda
# getting metadata for keypoint detection and deleting thing_classes to set to custom
metadata = MetadataCatalog.get(cfg_key.DATASETS.TRAIN[0]); del metadata.thing_classes
# keypoint detection person is 0 and weights are 1, so setting custom labels to lifter and weights
metadata.thing_classes = ['lifter','weights']
#%% 
def keypoint_degree(a,b,c):
  ba = a - b
  bc = c-b 
  cosine_angle = np.dot(ba,bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
  angle = np.arccos(cosine_angle)
  return np.degrees(angle)
# %%
def detect(videoPath):
  # get video and FPS
  video = cv2.VideoCapture(videoPath)
  width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) # width of frames in video stream
  height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height of frames in video stream
  fps = round(video.get(cv2.CAP_PROP_FPS)) # frame rate in video
  num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # number of frames in video
  # initializing videowriter
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 4-byte code used to specify the video codec
  video_writer = cv2.VideoWriter("test_video/output.mp4", fourcc, fps=float(fps), frameSize=(width, height), isColor=True)
  v = VideoVisualizer(metadata, ColorMode.IMAGE)
  pred_obj = DefaultPredictor(cfg_obj)
  pred_keypt = DefaultPredictor(cfg_key)
  def lift_finder(start_frame, end_frame, find_start=False, find_max_y=False):
    """ Function to help find the frame where the lift starts, ends, 
        and max height of the bar during the lift. Uses object detection."""
    if find_max_y:
      range_frames=range(start_frame, end_frame)
    else:
      # iterate through video every second
      range_frames = range(start_frame, end_frame, fps)
    for i, frame_num in enumerate(range_frames):
      # obtain frame
      video.set(cv2.CAP_PROP_POS_FRAMES, frame_num) #0-based index of the frame to be decoded/captured next.
      ret, frame = video.read()
      # make predictions
      outputs = pred_obj(frame)
      # filtering by largest bounding box, which should mean object detected is closest to camera
      idx_max = outputs['instances'].pred_boxes.area().cpu().argmax().tolist()
      if i == 0:
        # getting y1, which is top left corner of bounding box
        y1 = outputs['instances'].pred_boxes[[idx_max]].tensor.cpu().numpy()[0][1]
        # get height to pixel value for weightlifting plate after start of lift is known
        if ~find_start:
          top_y_pixels = outputs['instances'].pred_boxes[[idx_max]].tensor.cpu().numpy()[0][1]
          bottom_y_pixels = outputs['instances'].pred_boxes[[idx_max]].tensor.cpu().numpy()[0][3]
          # global the variable for usage when calculating velocity later
          global height_to_pixel_meters
          # (meters) / pixels
          height_to_pixel_meters = (17.72/39.37) / (bottom_y_pixels - top_y_pixels)
      pred_boxes1 = outputs['instances'].pred_boxes[[idx_max]]
      # get new y
      y1_new = pred_boxes1.tensor.cpu().numpy()[0][1]
      # finding frame with max bar height before the catch
      if find_max_y:
        # if new y1 higher above prev y1, update y1 and frame
        if y1_new < y1:
          y1 = y1_new
          frame_prev = frame_num
        # if new y1 is lower than prev y1, then weight is moving down and we found frame of max y during lift
        elif y1_new > y1:
          return frame_prev
      else:
        # start of the lift would be where y-coordinates change drastically
        pct_change = (y1_new - y1) / y1
        # if y1 decreases more than 20%(object moves upwards), then we found our starting point
        if (pct_change < -0.2) & find_start: 
          # start should be previous value
          return frame_prev
        # once bar is starting to drop, found our end point
        elif (pct_change > 0) & ~find_start:
          # want to return the current frame, not previous frame for end
          return frame_prev
        else:
          # set new y and frame start
          y1 = y1_new
          frame_prev = frame_num
  lift_start_frame= lift_finder(0, num_frames, find_start=True)
  lift_end_frame = lift_finder(lift_start_frame, num_frames)
  foi = {'lift_start': lift_start_frame, 'lift_end_frame': lift_end_frame}
  foi_instances = {}
  def lift_segmenter():
    for frame_num in range(lift_start_frame,lift_end_frame):
      # obtain frame
      video.set(cv2.CAP_PROP_POS_FRAMES, frame_num) #0-based index of the frame to be decoded/captured next.
      ret, frame = video.read()
      # make predictions
      outputs = pred_keypt(frame)
      # keypoint detection only looks for humans, so don't need to filter out other objects
      # filtering by largest bounding box, which should mean object detected is closest to camera
      idx_max = outputs['instances'].pred_boxes.area().cpu().argmax().tolist()
      pred_boxes1 = outputs['instances'].pred_boxes[[idx_max]]
      pred_scores1 = outputs['instances'].scores[[idx_max]]
      pred_classes1 = outputs['instances'].pred_classes[[idx_max]]
      pred_keypoints1 = outputs['instances'].pred_keypoints[[idx_max]]
      #pred_keypoints_heatmaps1 = outputs['instances']pred_keypoint_heatmaps[[idx_max]]
      # Set new instances for only one human
      new_instances = detectron2.structures.Instances(image_size=(height, width))
      new_instances.set('pred_boxes', pred_boxes1)
      new_instances.set('scores', pred_scores1)
      new_instances.set('pred_classes', pred_classes1)
      new_instances.set('pred_keypoints', pred_keypoints1)
      keypts = pred_keypoints1.cpu().numpy()[0]
      # find initial pull
      if 'initial_pull' not in foi:
        # use an average of left and right wrist/knee score to get highest confidence movement analysis
        left_wristknee_score = [keypts[9][2], keypts[13][2]]
        left_score = np.average(left_wristknee_score)
        right_wristknee_score = [keypts[10][2], keypts[14][2]]
        right_score = np.average(right_wristknee_score)
        # if scores are equivalent, use side with highest score/confidence
        if left_score == right_score:
          left_score = max(left_wristknee_score)
          right_score = max(right_wristknee_score)
        elif left_score > right_score:
          wrist_y = keypts[9][1]
          knee_y = keypts[13][1]
        elif right_score > left_score:
          wrist_y = keypts[10][1]
          knee_y = keypts[14][1]
        # when wrist goes above the knee, that's the end of the initial pull
        if wrist_y < knee_y:
          foi['initial_pull'] = frame_num
          foi_instances['initial_pull'] = new_instances
        continue
      if ('initial_pull' in foi) & ('power_position' not in foi):
        left_wristhip_score = [keypts[9][2], keypts[11][2]]
        left_score = np.average(left_wristhip_score)
        right_wristhip_score = [keypts[10][2], keypts[12][2]]
        right_score = np.average(right_wristhip_score)
        # if scores are equivalent, use side with highest score/confidence
        if left_score == right_score:
          left_score = max(left_wristhip_score)
          right_score = max(right_wristhip_score)
        elif left_score > right_score:
          wrist_y = keypts[9][1]
          hip_y = keypts[11][1]
        elif right_score > left_score:
          wrist_y = keypts[10][1]
          hip_y = keypts[12][1]
        # when wrist goes above the knee, that's the end of the initial pull
        if wrist_y < hip_y:
          foi['power_position'] = frame_num
          # storing left leg angle and right leg angle to detect catch
          left_hipkneeankle_score = [keypts[11][2], keypts[13][2], keypts[15][2]]
          left_score = np.average(left_hipkneeankle_score)
          right_hipkneeankle_score = [keypts[12][2], keypts[14][2], keypts[16][2]]
          right_score = np.average(right_hipkneeankle_score)
          if left_score == right_score:
            left_score = max(left_hipkneeankle_score)
            right_score = max(right_hipkneeankle_score)
          # use left side if score is higher
          elif left_score > right_score:
            hip =  np.array(keypts[11][:2])
            knee = np.array(keypts[13][:2])
            ankle = np.array(keypts[15][:2])
          # use right side if score is higher
          elif right_score > left_score:
            hip = np.array(keypts[12][:2])
            knee = np.array(keypts[14][:2])
            ankle = np.array(keypts[16][:2])
          leg_degree = keypoint_degree(hip, knee, ankle)
        continue
      # last position to detect, just need other two positions to be known
      if ('initial_pull' in foi) & ('power_position' in foi):
        left_hipkneeankle_score = [keypts[11][2], keypts[13][2], keypts[15][2]]
        left_score = np.average(left_hipkneeankle_score)
        right_hipkneeankle_score = [keypts[12][2], keypts[14][2], keypts[16][2]]
        right_score = np.average(right_hipkneeankle_score)
        if left_score == right_score:
          left_score = max(left_hipkneeankle_score)
          right_score = max(right_hipkneeankle_score)
        # use left side if score is higher
        elif left_score > right_score:
          hip =  np.array(keypts[11][:2])
          knee = np.array(keypts[13][:2])
          ankle = np.array(keypts[15][:2])
        # use right side if score is higher
        elif right_score > left_score:
          hip = np.array(keypts[12][:2])
          knee = np.array(keypts[14][:2])
          ankle = np.array(keypts[16][:2])
        leg_degree_new = keypoint_degree(hip, knee, ankle)
        if leg_degree_new < leg_degree:
          leg_degree = leg_degree_new
          foi['catch'] = frame_num
    return foi
  frames_of_interest = lift_segmenter()
  frames_of_interest['max_bar_height'] = lift_finder(frames_of_interest['power_position'],
                                                       frames_of_interest['catch'],
                                                       find_max_y=True)
  print(frames_of_interest)
  def xy_coord(frames_of_interest):
        frame_coordinates_left = {}
        frame_coordinates_right = {}
        left_plate_area = []
        right_plate_area = []
        # determining which plate bounding box is more consistent based on area of bounding box
        for point, frame in frames_of_interest.items():
          # obtain frame
          video.set(cv2.CAP_PROP_POS_FRAMES, frame) #0-based index of the frame to be decoded/captured next.
          ret, frame = video.read()
          # make predictions
          outputs = pred_obj(frame)
          # find top 2 largest boxes, these should be the plates closest to the camera
          box_areas, indices = outputs['instances'].pred_boxes.area().cpu().topk(2)
          indices = indices.tolist()
          pred_boxes1 = outputs['instances'].pred_boxes[indices]
          # not all predicted bounding boxes for the same object are the same size.
          #print(point, ' area:', pred_boxes1.area())
          # Bounding box coordinates are [x1, y1, x2, y2]. (x1,y1) is top left corner, (x2,y2) bottom right corner
          # Objects may get detected in different orders, so will separate based on x,y values. Larger x-value is right
          # Using bottom right point as it has less risk of moving out of frame
          x2_0 = pred_boxes1.tensor.cpu().numpy()[0][2] # first plate bottom right x
          x2_1 = pred_boxes1.tensor.cpu().numpy()[1][2] # second plate bottom right x
          if x2_0 < x2_1: # if first box is for the left plate
            left_plate_xy = pred_boxes1.tensor.cpu().numpy()[0][2:]
            right_plate_xy = pred_boxes1.tensor.cpu().numpy()[1][2:]
            left_plate_area.append(box_areas.numpy()[0])
            right_plate_area.append(box_areas.numpy()[1])
          elif x2_0 > x2_1: # if first box is for the right plate
            left_plate_xy = pred_boxes1.tensor.cpu().numpy()[1][2:]
            right_plate_xy = pred_boxes1.tensor.cpu().numpy()[0][2:]
            left_plate_area.append(box_areas.numpy()[1])
            right_plate_area.append(box_areas.numpy()[0])
          frame_coordinates_left[point] = left_plate_xy.tolist()
          frame_coordinates_right[point] = right_plate_xy.tolist()
        # keep the xy coordinates of the plate bounding box with the most consistent area in all the frames
        if np.std(left_plate_area) <= np.std(right_plate_area):
          return frame_coordinates_left
        else:
          return frame_coordinates_right
  # calculate xy_coordinate outside of loop to save from running the function multiple times
  frame_coordinates = xy_coord(frames_of_interest)
  def lift_velocity(frames_coordinates):
    lift_points = ['lift_start','initial_pull','power_position', 'max_bar_height','catch','lift_end_frame']
    velocities = {}
    for point1, point2 in zip(lift_points, lift_points[1::]):
      x_pt1, y_pt1 = frame_coordinates[point1]
      x_pt2, y_pt2 = frame_coordinates[point2]
      # find degree of motion(not needed yet)
      # radians= math.atan2(y_pt2 - y_pt1, x_pt2 - x_pt1)
      # degrees= math.degrees(radians_left)
      # calculate time
      time_s = (frames_of_interest[point2] - frames_of_interest[point1]) / fps
      # calculate velocity components for left and right weightlifting plates
      vx = (x_pt2 - x_pt1) / time_s * height_to_pixel_meters # pixels/s * meters/pixels
      vy = (y_pt2 - y_pt1) / time_s * height_to_pixel_meters
      # velocity is sqrt of sum of velocity components squared
      v_tot= math.sqrt(vx**2 + vy**2) # velocity is in meters/s
      # average velocities from left and right plates to get approximate velocity in m/s
      velocities[point1 + '_to_' + point2] = round(np.mean([v_tot, v_tot]), 2)
      # need to calculate max y before catch
    return velocities
  velocities = lift_velocity(frames_of_interest)
  print(velocities)
  
  def image_writer(predictor, metadata, frame_of_interest, directory, custom_prefix):
      # obtain frame
      video.set(cv2.CAP_PROP_POS_FRAMES, frame_of_interest) #0-based index of the frame to be decoded/captured next.
      ret, frame = video.read()
      # make predictions
      outputs = predictor(frame)
      v = Visualizer(frame[:, :, ::-1], metadata, scale=1.2)
      out = v.draw_instance_predictions(outputs['instances'].to("cpu"))
      cv2.imwrite(f'{directory}/{custom_prefix}_{frame_of_interest}.jpg', out.get_image())
  # for point in frames_of_interest.keys():
  #   image_writer(pred_obj, MetadataCatalog.get("barbell_train"),
  #                frames_of_interest[point], 'barbell_frames', point)
  def video_writer(video):
    for n_frame in range(lift_start_frame, lift_end_frame):
      # obtain frame
      video.set(cv2.CAP_PROP_POS_FRAMES, n_frame) #0-based index of the frame to be decoded/captured next.
      ret, frame = video.read()
      # make predictions
      outputs_key = pred_keypt(frame)
      idx_max = outputs_key['instances'].pred_boxes.area().cpu().argmax().tolist()
      pred_boxes_key = outputs_key['instances'].pred_boxes[[idx_max]]
      pred_scores_key = outputs_key['instances'].scores[[idx_max]]
      pred_classes_key = outputs_key['instances'].pred_classes[[idx_max]]
      pred_keypoints = outputs_key['instances'].pred_keypoints[[idx_max]]
      #pred_keypoints_heatmaps = outputs['instances'].pred_keypoint_heatmaps[[idx_max]]
      outputs_obj = pred_obj(frame)
      # find top 2 largest boxes, these should be the plates closest to the camera
      val, indices = outputs_obj['instances'].pred_boxes.area().cpu().topk(2)
      pred_boxes_obj = outputs_obj['instances'].pred_boxes[[indices]]
      pred_scores_obj = outputs_obj['instances'].scores[[indices]]
      pred_classes_obj = outputs_obj['instances'].pred_classes[[indices]]
      # Set new instances for lifter and weight
      new_instances = detectron2.structures.Instances(image_size=(height, width))
      new_instances.set('pred_boxes', Boxes.cat([pred_boxes_key, pred_boxes_obj]))
      new_instances.set('scores', torch.cat([pred_scores_key, pred_scores_obj]))
      new_instances.set('pred_classes', torch.cat([pred_classes_key, pred_classes_obj]))
      new_instances.set('pred_keypoints', torch.cat([pred_keypoints, pred_keypoints, pred_keypoints]))
      # Make sure the frame is colored
      frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
      visualization = v.draw_instance_predictions(frame, new_instances.to("cpu"))
      # Convert Matplotlib RGB format to OpenCV BGR format
      visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
      cv2.imwrite(f'test_video/{n_frame}.jpg', visualization)
      video_writer.write(visualization)
    video.release()
    video_writer.release()
    cv2.destroyAllWindows()
  video_writer(video)
    # write object and keypoint bounding boxes to video if possible
    # next steps: get max bar height, get bar height during catch, calculate hook, form recommendations
    # use weightlifting plate with most consistent bounding box area through out all positiosn of the lift to calculate items
#%%
outputs = detect(videoPath)
# %% Keypoint indices
keypoint_names = metadata.keypoint_names
keypoint_names.index('nose')
# "keypoints": [
# "nose",
# "left_eye",
# "right_eye",
# "left_ear",
# "right_ear",
# "left_shoulder",
# "right_shoulder",
# "left_elbow",
# "right_elbow",
# "left_wrist",
# "right_wrist",
# "left_hip",
# "right_hip",
# "left_knee",
# "right_knee",
# "left_ankle",
# "right_ankle"
# ]