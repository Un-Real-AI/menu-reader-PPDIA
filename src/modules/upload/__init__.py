import boto3
import botocore
from flask import jsonify
import uuid

def random_name(s3_client,bucket_name,file_extension):
    file_name = str(uuid.uuid4())
    try: 
        s3_client.Object(bucket_name, file_name+file_extension).load()
    except:
        return file_name
    return random_name(s3_client,bucket_name,file_extension)

def sent_to_s3(file, bucket_name):
    file_extension = file.filename.split(".")[-1]
    if file_extension not in ['jpg','png','pdf','jpeg','webp']:
        return jsonify(process_id=None,state=0,message="The file is not a supported format. Please upload a .png, .jpg, .pdf, .jpeg or .webp file")
    s3_client = boto3.client('s3')
    file_name= random_name(s3_client,bucket_name, file_extension)
    try:
        s3_client.upload_fileobj(
            file,
            bucket_name,
            file_name+'.'+file_extension,
            ExtraArgs={
                "ContentType": file.content_type    #Set appropriate content type as per the file
            }
        )
    except Exception as e:
        print("Something Happened: ", e)
        return e,file_name+'.'+file_extension
    return jsonify(process_id=file_name,state=1,message="Successfully upload"),file_name+'.'+file_extension

def check_status(process_id,bucket_name):
    s3 = boto3.resource('s3')
    try:
        print(process_id+'.json')
        s3.Object(bucket_name, process_id+'.json').load()
        return jsonify(status=1,message="Process: {} finished, ready to query".format(process_id))
    except botocore.exceptions.ClientError as e:

        if e.response['Error']['Code'] == "404":
            # The object does not exist.
            s3_client = boto3.client('s3')
            objs = s3_client.list_objects_v2(Bucket=bucket_name,Prefix=process_id)['Contents']
            if len(objs) == 0:
                return jsonify(status=0,message="Process: {} never created".format(process_id))
            else:
                return jsonify(status=0,message="Process: {} created, in process".format(process_id))
        else: 
            raise e
