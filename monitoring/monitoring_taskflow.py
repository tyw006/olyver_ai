# Importing Libraries
import pandas as pd
import json
import boto3
import uuid
import os
# from dotenv import load_dotenv
from datetime import datetime, timedelta 
from airflow.decorators import task, dag
from arize.pandas.logger import Client
from arize.utils.types import (
    Schema, 
    Environments, 
    ModelTypes,
    EmbeddingColumnNames,
    ObjectDetectionColumnNames
)

# load_dotenv()
@dag(
    schedule=timedelta(minutes=1),
    start_date=datetime(2024,1,27),
    catchup=False
)
def monitoring_taskflow():
    """Main function for running Airflow Tasks."""
    @task
    def data_prep(data):
        """ This function is to be ran on a schedule defined by airflow.
        It will sample a small percentage of data from our testing set.
        Testing set should be located in the same directory."""
        df = pd.read_parquet(data)
        # Sending each test image to sagemaker for inference
        test_sample = df.sample(n=50, ignore_index=True)
        # Initialize new columns
        test_sample['pred_category'] = [None] * len(test_sample)
        test_sample['pred_bbox'] = [None] * len(test_sample)
        test_sample['pred_score'] = [None] * len(test_sample)
        for idx, jpeg_bytes in enumerate(test_sample.img_bytes):
            sagemaker_client = boto3.client('sagemaker-runtime')
            response = sagemaker_client.invoke_endpoint(
                EndpointName='d2-serve',
                Body=jpeg_bytes,
                ContentType='image/jpeg',
                Accept="json"
            )
            response_dict = response['Body'].read()
            pred_dict = json.loads(response_dict)
            # Actual bboxes contain only one detected object
            # Keep only highest scoring object
            # idxmax = pred_dict['scores'].index(max(pred_dict['scores']))
            test_sample.at[idx, 'pred_category'] = pred_dict['pred_classes']#[idxmax]
            test_sample.at[idx, 'pred_bbox'] = pred_dict['pred_boxes']#[idxmax]]
            test_sample.at[idx, 'pred_score'] = pred_dict['scores']#[idxmax]]
        # Remove undetected samples as model may not be trained to detect object
        undetect = test_sample.pred_bbox.apply(lambda x: len(x) == 0)
        test_sample.drop(test_sample[undetect].index, inplace=True)
        test_sample.reset_index(inplace=True, drop=True)
        # Standardize categories
        # lambda function used to correct categories below
        correct_cat = lambda x,y: ['Barbell'] * len(y) if x == 1 else ['barbell'] * len(y)
        test_sample['pred_category'] = test_sample.apply(
            lambda row: correct_cat(row['pred_category'], row['pred_bbox']),axis=1)
        test_sample['actual_category'] = test_sample.apply(
            lambda row: correct_cat(row['actual_category'], row['actual_bbox']),axis=1)
        # Set timestamp to now
        test_sample['prediction_ts'] = datetime.timestamp(datetime.now())

        # Adding prediction ids
        def add_prediction_id(df):
            return [str(uuid.uuid4()) for _ in range(df.shape[0])]
        test_sample['prediction_id'] = add_prediction_id(test_sample)

        return test_sample
    
    @task
    def arize_push(test_sample):
        # Sending data to Arize

        SPACE_KEY = #Insert space key here
        API_KEY = #Insert API key here
        arize_client = Client(space_key=SPACE_KEY, api_key=API_KEY)
        model_id = "detectron2-barbell-detection"
        model_version = "1.0"
        model_type = ModelTypes.OBJECT_DETECTION
        if SPACE_KEY == "SPACE_KEY" or API_KEY == "API_KEY":
            raise ValueError("❌ NEED TO CHANGE SPACE AND/OR API_KEY")
        else:
            print("✅ Import and Setup Arize Client Done! Now we can start using Arize!")

        embedding_feature_column_names={
            "image_embedding": EmbeddingColumnNames(
                vector_column_name="image_vector"
            )
        }
        object_detection_prediction_column_names=ObjectDetectionColumnNames(
            bounding_boxes_coordinates_column_name="pred_bbox",
            categories_column_name="pred_category",
            scores_column_name="pred_score"
        )
        object_detection_actual_column_names=ObjectDetectionColumnNames(
            bounding_boxes_coordinates_column_name="actual_bbox",
            categories_column_name="actual_category",
        )

        # Define a Schema() object for Arize to pick up data from the correct columns for logging
        schema = Schema(
            prediction_id_column_name="prediction_id",
            timestamp_column_name="prediction_ts",
            #tag_column_names=tags,
            embedding_feature_column_names=embedding_feature_column_names,
            object_detection_prediction_column_names=object_detection_prediction_column_names,
            object_detection_actual_column_names=object_detection_actual_column_names,
        )

        # Logging DataFrame
        response = arize_client.log(
            dataframe=test_sample,
            schema=schema,
            model_id=model_id,
            model_version=model_version,
            model_type=model_type,
            environment=Environments.PRODUCTION,
        )

        # If successful, the server will return a status_code of 200
        if response.status_code != 200:
            print(f"❌ logging failed with response code {response.status_code}, {response.text}")
        else:
            print(f"✅ You have successfully logged training set to Arize")
    
    test_sample = data_prep()# insert link to test dataframe as argument
    arize_push(test_sample)

monitoring_taskflow()