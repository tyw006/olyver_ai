import streamlit as st
import cv2
import requests
import base64
import json
import numpy as np
from PIL import Image
import torch
from detectron2.structures import Instances, Boxes
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode
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

def json_to_d2(pred_dict, device):
    """ 
    Client side helper function to deserialize the JSON msg back to d2 outputs 
    """
    
    pred_dict = json.loads(pred_dict)
    for k, v in pred_dict.items():
        if k=="pred_boxes":
            boxes_to_tensor = torch.FloatTensor(v).to(device)
            pred_dict[k] = Boxes(boxes_to_tensor)
        if k=="scores":
            pred_dict[k] = torch.Tensor(v).to(device)
        if k=="pred_classes":
            pred_dict[k] = torch.Tensor(v).to(device).to(torch.uint8)
    
    height, width = pred_dict['image_size']
    del pred_dict['image_size']

    inst = Instances((height, width,), **pred_dict)
    
    return {'instances':inst}

def process_uploaded_file(uploaded_file):
    # Check if a file is uploaded
    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[0]
        input_img = np.array(Image.open(uploaded_file))
        if file_type == 'image':
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
            # Display the uploaded video
            st.video(uploaded_file)

            # Process the video (you can add more video processing logic here)
            # For simplicity, this example doesn't include video processing.

        else:
            st.warning("Unsupported file type. Please upload an image or video.")

# Streamlit app
def main():
    st.title("File Uploader App")

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png", "gif", "mp4"])

    # Process the uploaded file
    process_uploaded_file(uploaded_file)

if __name__ == "__main__":
    main()
