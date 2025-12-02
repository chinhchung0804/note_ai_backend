import os
import boto3
from botocore.client import Config

S3_ENDPOINT = os.getenv('S3_ENDPOINT_URL')
S3_ACCESS = os.getenv('S3_ACCESS_KEY')
S3_SECRET = os.getenv('S3_SECRET_KEY')
S3_BUCKET = os.getenv('S3_BUCKET')

def upload_file(path, key):
    if not S3_ENDPOINT:
        return None
    s3 = boto3.resource('s3',
                        endpoint_url=S3_ENDPOINT,
                        aws_access_key_id=S3_ACCESS,
                        aws_secret_access_key=S3_SECRET,
                        config=Config(signature_version='s3v4'))
    s3.Bucket(S3_BUCKET).upload_file(path, key)
    return f'{S3_ENDPOINT}/{S3_BUCKET}/{key}'
