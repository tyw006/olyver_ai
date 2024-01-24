# Change directory to deploy-sagemaker
directory="deploy-sagemaker"
cd $directory

# zip the code and model together
tar_file_name='model.tar.gz'
tar -czvf $tar_file_name *

echo "Model successfully compressed."

# Upload to s3. It will replace file if exists.
S3_BUCKET='detectron2-weightlifting'
S3_MODEL_PATH="s3://$S3_BUCKET/$tar_file_name"
aws s3 cp $tar_file_name $S3_MODEL_PATH

echo "Model successfully uploaded to S3."

# Delete tar file so it won't get repackaged next time
rm $tar_file_name

echo "$tar_file_name successfully removed from $directory."

# Logging into docker & shared ECR repo. 
# Need to do this to get image from aws and push docker image to aws
echo "Logging into AWS Elastic Container Registry."

aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-2.amazonaws.com
# logging into private ECR
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 187135372127.dkr.ecr.us-east-2.amazonaws.com

# Run build and push bash script. Takes 3 arguments: image name, image tag, dockerfile.
cd ..
source build_and_push_image.sh sagemaker-d2-train-weightlifting latest Dockerfile.serving us-east-2

echo "Image pushed to ECR."