import streamlit as st
import cv2
import requests
import base64
import json
import numpy as np
from PIL import Image
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from olympic_detection import AWSDetector
from olympic_detection import Detector
from olympic_detection import json_to_d2
import json 
import os

dataset_name = "test"
dataset_location = "s3://detectron2-weightlifting/detectron2/data/test/"
annotation_file = "_annotations.coco.json"
image_dir = "Barbell-small-2\test"
if not 'test' in DatasetCatalog.list():
    register_coco_instances(dataset_name, {}, os.path.join(dataset_location, annotation_file), 
                            os.path.join(dataset_location, image_dir))
cb_meta = MetadataCatalog.get(dataset_name); #del cb_meta.thing_classes
MetadataCatalog.get('test').thing_classes = ['Lifter', 'Olympic Plate']

def process_uploaded_file(uploaded_file):
    # Check if a file is uploaded
    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        # image capability was for testing functionality of model on AWS
        if file_type == 'image':
            input_img = np.array(Image.open(uploaded_file))
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
            # Uploaded Image should be a bytes type object. Need to convert to b64.
            # API Gateway has issues handling byte objects.
            image_b64 = base64.b64encode(uploaded_file.read()).decode("utf-8")
            # Send to API Gateway/Lambda function
            api_url = 'https://a9gt582gfk.execute-api.us-east-2.amazonaws.com/deploy/d2-weightlifting-predict'
            headers = {'Content-Type': 'image/jpeg'}
            data = json.dumps({"body": image_b64})
            response = requests.post(api_url, data=data, headers=headers)
            st.write(response)
            # response.text should be a JSON
            predictions = json_to_d2(response.text, 'cpu')
            st.image(input_img)
            v = Visualizer(input_img, # Visualizer reads image numpy array
                            metadata=cb_meta,
                            scale=1.2)
            out = v.draw_instance_predictions(predictions["instances"].to("cpu"))
            # drawing instance predictions reverses the color channel order, so need to revert with [:, :, ::-1]
            image_final = cv2.cvtColor(out.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
            st.image(image_final, use_column_width=True)
        elif file_type == 'video':
            # For Local
            if 'velocities' not in st.session_state:
                st.session_state['velocities'] = Detector().video_detect(uploaded_file)
                # For AWS
                # st.session_state['velocities'] = AWSDetector().video_detect(uploaded_file)
            st.subheader("Velocities per segment of the lift: \n")
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Initial Pull", 
                        f"{st.session_state.velocities['lift_start_to_initial_pull']} m/s")
            col2.metric("Initial Pull to Power Position", 
                        f"{st.session_state.velocities['initial_pull_to_power_position']} m/s")
            col3.metric("Power Position to Max Bar Height", 
                        f"{st.session_state.velocities['power_position_to_max_bar_height']} m/s")
            col4.metric("Max Bar Height to Catch", 
                        f"{st.session_state.velocities['max_bar_height_to_catch']} m/s")
            col5.metric("Catch to Lift End", 
                        f"{st.session_state.velocities['catch_to_lift_end']} m/s")
        else:
            st.warning("Unsupported file type. Please upload an image or video.")

# Streamlit app
def main():
    st.set_page_config(layout='wide')
    _col1, column, _col2 = st.columns([1, 5, 1])
    with column:
        st.title("Olympic Lifting Evaluator")
        st.caption("Olympic lifting is a highly technical movement that requires strength, speed, and explosive power. "\
                    "This application uses Detectron2 to use both keypoint and object detection to measure the speed \
                        of the lift in each portion of the lift. The future goal of this application would be to evaluate "\
                            "each of their lifts in real time, even including coaching tips.")
        st.write('\n\n')
        st.caption("Begin by uploading an mp4 file of a lift. The application will select the first lift that appears.")
        # File uploader. Currently only accept video.
        uploaded_file = st.file_uploader("Choose a file", type=["mp4",
                                                                #"jpg", "jpeg", "png", "gif", "mp4"
                                                                ]
                                                                )

        # Process the uploaded file
        process_uploaded_file(uploaded_file)

if __name__ == "__main__":
    main()
