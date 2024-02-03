from detectron2.engine import DefaultPredictor
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import Boxes
from olympic_detection import json_to_d2

import streamlit as st
from PIL import Image
import cv2
from dotenv import load_dotenv
import numpy as np
import math
import torch
import os
import pathlib

class Detector:

    def __init__(self):
        load_dotenv()
        WEIGHTS = os.getenv("LOCAL_WEIGHTS")
        # Defining object detector
        self.cfg_obj = get_cfg() # obtain detectron2's default config
        self.cfg_obj.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
        self.cfg_obj.DATASETS.TEST = ("barbell_val",)
        self.cfg_obj.MODEL.ROI_HEADS.NUM_CLASSES = 2
        self.cfg_obj.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
        self.cfg_obj.MODEL.WEIGHTS = WEIGHTS #pre-trained weights on local, stored in folder
        #Defining keypoint detector
        self.cfg_key = get_cfg() # obtain detectron2's default config
        # may need to download config file locally for greater speed
        self.cfg_key.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")) # load values from a file specified under Detectron2's official configs/ directory
        self.cfg_key.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml") # returns the URL to the model trained using the given config
        self.cfg_key.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7 # threshold to filter out low-scored bounding boxes
        self.cfg_key.MODEL.DEVICE = 'cuda'  # cpu or cuda
        # getting metadata for keypoint detection and deleting thing_classes to set to custom
        self.metadata = MetadataCatalog.get(self.cfg_key.DATASETS.TRAIN[0]); del self.metadata.thing_classes
        # keypoint detection person is 0 and weights are 1, so setting custom labels to lifter and weights
        self.metadata.thing_classes = ['lifter','weights']
        # Defining the predictors
        self.pred_obj = DefaultPredictor(self.cfg_obj)
        self.pred_keypt = DefaultPredictor(self.cfg_key)
    
    def video_detect(self, uploaded_video):
        # Write video to temp dir. CV2 only accepts video files
        STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
        VIDEOS_PATH = (STREAMLIT_STATIC_PATH / "videos")
        if not VIDEOS_PATH.is_dir():
            VIDEOS_PATH.mkdir()
        os.makedirs(f'{VIDEOS_PATH}/VIDEOS_PATH_temp', exist_ok=True)
        temp_file_path = os.path.join(f"{VIDEOS_PATH}/VIDEOS_PATH_temp", "input_video.mp4")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_video.read())
        # get video and FPS
        video = cv2.VideoCapture(temp_file_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)) # width of frames in video stream
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) #height of frames in video stream
        fps = round(video.get(cv2.CAP_PROP_FPS)) # frame rate in video
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # number of frames in video
        # initializing videowriter
        fourcc = cv2.VideoWriter_fourcc(*'H264') # streamlit video doesn't support mp4v
        video_writer = cv2.VideoWriter(f"{VIDEOS_PATH}/output.mp4", fourcc, fps=float(fps), frameSize=(width, height), isColor=True)
        v = VideoVisualizer(self.metadata, ColorMode.IMAGE)

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
                outputs = self.pred_obj(frame)
                # outputs = self.pred_obj(frame)
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
                        # global height_to_pixel_meters
                        # (meters) / pixels
                        self.height_to_pixel_meters = (17.72/39.37) / (bottom_y_pixels - top_y_pixels)
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
        foi = {'lift_start': lift_start_frame,
               'lift_end': lift_end_frame}
        def keypoint_degree(a,b,c):
            """ Function to calculate angle between three points in space."""
            ba = a - b
            bc = c-b 
            cosine_angle = np.dot(ba,bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            angle = np.arccos(cosine_angle)
            return np.degrees(angle)
        foi_instances={}
        def lift_segmenter():
            for frame_num in range(lift_start_frame,lift_end_frame):
                # obtain frame
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_num) #0-based index of the frame to be decoded/captured next.
                ret, frame = video.read()
                # make predictions
                outputs = self.pred_keypt(frame)
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
        def xy_coord(frames_of_interest):
            """ Function for determining which weightlifting plate bbox coordinate to use.
            Which ever bbox has the least amount of change over all the frames will be selected."""
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
                outputs = self.pred_obj(frame)
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
        def lift_velocity(frame_coordinates):
            lift_points = ['lift_start','initial_pull','power_position',
                           'max_bar_height','catch','lift_end']
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
                vx = (x_pt2 - x_pt1) / time_s * self.height_to_pixel_meters # pixels/s * meters/pixels
                vy = (y_pt2 - y_pt1) / time_s * self.height_to_pixel_meters
                # velocity is sqrt of sum of velocity components squared
                v_tot= math.sqrt(vx**2 + vy**2) # velocity is in meters/s
                # average velocities from left and right plates to get approximate velocity in m/s
                velocities[point1 + '_to_' + point2] = round(np.mean([v_tot, v_tot]), 2)
                # need to calculate max y before catch
            return velocities
        velocities = lift_velocity(frame_coordinates)
        def video_output(video):
            for n_frame in range(lift_start_frame, lift_end_frame):
                # obtain frame
                video.set(cv2.CAP_PROP_POS_FRAMES, n_frame) #0-based index of the frame to be decoded/captured next.
                ret, frame = video.read()
                # make predictions
                outputs_key = self.pred_keypt(frame)
                idx_max = outputs_key['instances'].pred_boxes.area().cpu().argmax().tolist()
                pred_boxes_key = outputs_key['instances'].pred_boxes[[idx_max]]
                pred_scores_key = outputs_key['instances'].scores[[idx_max]]
                pred_classes_key = outputs_key['instances'].pred_classes[[idx_max]]
                pred_keypoints = outputs_key['instances'].pred_keypoints[[idx_max]]
                #pred_keypoints_heatmaps = outputs['instances'].pred_keypoint_heatmaps[[idx_max]]
                outputs_obj = self.pred_obj(frame)
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
                # cv2.imwrite(f'{temp_dir}/{n_frame}.jpg', visualization)
                video_writer.write(visualization)
            video.release()
            video_writer.release()
            cv2.destroyAllWindows()
        video_output(video)
        # video_file = open(f'{VIDEOS_PATH}\output.mp4', 'rb')
        # video_bytes = video_file.read()
        # st.video(video_bytes)
        st.video(f'{VIDEOS_PATH}/output.mp4')
        # Cleaning up files and directory
        os.remove(temp_file_path)
        os.rmdir(rf"{VIDEOS_PATH}\VIDEOS_PATH_temp")
        os.remove(f'{VIDEOS_PATH}/output.mp4')
        return velocities