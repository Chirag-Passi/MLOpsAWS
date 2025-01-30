import boto3

def approve_latest_model_package(model_package_group_name, region='us-east-1'):
    # Create a SageMaker client
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    try:
        # Fetch the latest model package version
        response = sagemaker_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            SortBy='CreationTime',
            MaxResults=1,
            SortOrder='Descending'  # Get the latest version
        )
        
        if 'ModelPackageSummaryList' in response and len(response['ModelPackageSummaryList']) > 0:
            # Get the latest model package
            latest_model_package = response['ModelPackageSummaryList'][0]
            latest_model_package_arn = latest_model_package['ModelPackageArn']
            
            # Approve the latest model package using ModelPackageArn
            sagemaker_client.update_model_package(
                ModelPackageArn=latest_model_package_arn,
                ModelApprovalStatus='Approved'
            )
            
            print(f"Approved model package {latest_model_package_arn}.")
            return latest_model_package_arn
        else:
            print(f"No model package found in group {model_package_group_name}.")
            return None

    except Exception as e:
        print(f"Error approving model package: {str(e)}")
        return None

# Call the function
approve_latest_model_package('MLOpsAWSModelGroup')
