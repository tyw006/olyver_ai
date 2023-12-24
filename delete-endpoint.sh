endpoint_name="d2-serve"

# delete sagemaker endpoint
aws sagemaker delete-endpoint --endpoint-name $endpoint_name

# delete sagemaker endpoint config
aws sagemaker delete-endpoint-config --endpoint-config-name "$endpoint_name"

echo "endpoint '$endpoint_name' and endpoint configuration deleted."
