import os
import boto3
import json
import logging 
import base64

logger = logging.getLogger()
logger.setLevel(logging.INFO)
# grab environment variables
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    """ Returns predictions dictionary that includes boxes, scores,
    and predicted classes tensor."""

    logger.info("Received event: " + json.dumps(event, indent=2))
    try:
        payload = base64.b64decode(event['body'])
        logger.info('Invoking Endpoint...')
        response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                        ContentType='image/jpeg',
                                        Body=payload,
                                        Accept="json")
        response_dict= response['Body'].read()
        logger.info("Predictions complete!")
        logger.info(response_dict)
        return response_dict
        
    except Exception as e:
        logger.error(e)