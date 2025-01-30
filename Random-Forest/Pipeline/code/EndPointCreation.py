import boto3
import time

def deploy_latest_model_version(model_group_name, endpoint_name, execution_role_arn, instance_type='ml.m5.xlarge', instance_count=1):
    """
    Deploy the latest approved model version from a SageMaker Model Group to an endpoint.
    
    Args:
        model_group_name (str): Name of the model group
        endpoint_name (str): Name for the endpoint to create/update
        execution_role_arn (str): ARN of the IAM role for SageMaker execution
        instance_type (str): ML instance type for deployment
        instance_count (int): Number of instances to deploy
    """
    sm_client = boto3.client('sagemaker')
    
    # Get the latest approved model version
    response = sm_client.list_model_packages(
        ModelPackageGroupName=model_group_name,
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )
    
    if not response['ModelPackageSummaryList']:
        raise Exception(f"No approved model versions found in group {model_group_name}")
    
    model_package_arn = response['ModelPackageSummaryList'][0]['ModelPackageArn']
    
    # Create model
    model_name = f"{model_group_name}-{int(time.time())}"
    sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=execution_role_arn,
        PrimaryContainer={
            'ModelPackageName': model_package_arn
        }
    )
    
    # Check if endpoint exists
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
    except sm_client.exceptions.ClientError:
        endpoint_exists = False
    
    # Create endpoint configuration
    endpoint_config_name = f"{endpoint_name}-config-{int(time.time())}"
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[{
            'VariantName': 'AllTraffic',
            'ModelName': model_name,
            'InstanceType': instance_type,
            'InitialInstanceCount': instance_count,
            'InitialVariantWeight': 1
        }]
    )
    
    if endpoint_exists:
        # Update existing endpoint
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    else:
        # Create new endpoint
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    
    print(f"Waiting for endpoint {endpoint_name} to be ready...")
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(EndpointName=endpoint_name)
    print(f"Endpoint {endpoint_name} is ready!")

# Example usage
if __name__ == "__main__":
    # You need to provide your SageMaker execution role ARN
    execution_role = "arn:aws:iam::767397996001:role/SageMakerMLOps"
    
    deploy_latest_model_version(
        model_group_name='MLOpsAWSModelGroup',
        endpoint_name='my-endpoint',
        execution_role_arn=execution_role,
        instance_type='ml.m5.large',
        instance_count=1
    )