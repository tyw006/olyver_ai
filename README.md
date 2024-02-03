# Olympic Lifting Evaluator

Olympic Lifting Evaluator is a computer vision application that analyzes Olympic lifting videos to calculate velocities for the barbell during different phases of the lift.

## Overview

Olympic lifting is a highly technical movement that demands strength, speed, and explosive power. This application utilizes Detectron2 for both keypoint and object detection to measure the speed of the lift in various portions. The goal is to evaluate each lift in real-time, providing valuable insights and even coaching tips.

The application utilizes a locally trained Detectron2 model for olympic plate detection. The model is then pushed and deployed to Amazon Web Services(S3, Sagemaker). A streamlit front end is used for user interfacing. Upon upload of a lifting video, a request will be sent through Amazon API Gateway, then Amazon Lambda, which will then invoke the endpoint of the model. The predictions are sent back to the application and displayed underneath the final video of the lifter with the detection overlayed. See overview diagram:

![Diagram](https://github.com/tyw006/olyver_ai/blob/main/images/Technical%20Diagrams.jpg)


## Features

- Key and object detection using Detectron2.
- Calculation of velocities for different phases of the Olympic lift.
- Supports mp4 video input format.

## How to Use

1. **Upload a Video**: Begin by uploading an MP4 file containing an Olympic lifting sequence. The application will automatically select the first lift that appears.

    ![UploadImage](https://github.com/tyw006/olyver_ai/blob/main/images/app_input.png)

2. **Analysis Results**: The application will process the uploaded video, displaying velocities for each segment of the lift. The results include metrics such as Initial Pull, Power Position, Max Bar Height, Catch, and Lift End.

    ![ResultsGIF](https://github.com/tyw006/olyver_ai/blob/main/images/output.gif)
    ![Metrics](https://github.com/tyw006/olyver_ai/blob/main/images/output_metrics1.png)

3. **Monitoring Model**: An Airflow task scheduler is used to send test data to the endpoint on a periodic basis to monitor accuracy of the model. It is important to ensure that the model continues to detect the objects without any drift. The endpoint data is the sent to Arize for monitoring and visualization.

    ![Monitoring](https://github.com/tyw006/olyver_ai/blob/main/images/ArizeMonitoring.png)

## Getting Started

Follow these steps to run the application locally:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/olympic-lifting-evaluator.git
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
3. Make sure model and image are pushed to S3 and ECR, respectively:
    ```
    bash push_model_image_to_aws.sh
4. Deploy model on AWS Sagemaker:
    ```
    bash deploy-realtime.sh
5. Start streamlit application:
    ```
    streamlit run app.py
6. When finished, delete endpoint to save costs:
    ```
    bash delete-endpoint.sh