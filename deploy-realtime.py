import sagemaker
from sagemaker.pytorch import PyTorchModel

bucket = 'detectron2-weightlifting'
role = "arn:aws:iam::187135372127:role/SageMakerFullAccess"
region = 'us-east-2'
sm_session = sagemaker.Session(default_bucket=bucket)
account = sm_session.account_id()
# Model should have been uploaded previously to S3
model_url = "s3://{}/{}".format(bucket, "model.tar.gz")
# Image should have been uploaded previously to ECR
serve_container_name = 'sagemaker-d2-serve-weightlifting'
serve_container_version = 'latest'
serve_image_uri = f"{account}.dkr.ecr.{region}.amazonaws.com/{serve_container_name}:{serve_container_version}"
# Creating inference model
remote_model=PyTorchModel(name='d2-deploy',
                         model_data=model_url,
                         role=role,
                         sagemaker_session=sm_session,
                         entry_point="inference.py",
                         image_uri=serve_image_uri,
                         framework_version="1.6.0",
                         py_version='py3')
# Deploying model
endpoint_name = 'd2-serve'
remote_predictor = remote_model.deploy(
                         instance_type='ml.g4dn.xlarge', # has GPU - EC2 instance type to deploy model to
                        #  instance_type = 'ml.m5.xlarge', # no GPU
                         initial_instance_count=1, # number of instances to run in endpoint
                         endpoint_name=endpoint_name, # define a unique endpoint name; if ommited, Sagemaker will generate it based on used container
                         wait=True, # wait for deployment of model completes to call             
                         )
print("\nSuccessfully deployed to Realtime Endpoint: ", remote_predictor.endpoint_name)