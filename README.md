# olyver_ai

Model is pre-trained locally and will be pushed to AWS along with the image required to run the model. 

1. Run bash script push_model_image_to_aws.sh to push the model to s3 bucket and image to ECR. Only run if changes were made to the model.

2. Run "python deploy-realtime.py" to deploy the model on AWS Sagemaker. 

3. Start streamlit application using "streamlit run app.py"

4. When finished, run delete-endpoint.sh to remove the endpoint and save costs.st