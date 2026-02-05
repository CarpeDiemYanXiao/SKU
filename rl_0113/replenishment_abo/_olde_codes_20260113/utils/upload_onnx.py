import os
import time
import zipfile
import boto3
import subprocess


def zip_files(file1, file2, zip_path):
    with zipfile.ZipFile(zip_path, "w") as zipf:
        zipf.write(file1, arcname=os.path.basename(file1))
        zipf.write(file2, arcname=os.path.basename(file2))


def upload_to_s3(zip_path, bucket, key):
    uKey = "sls_mart"
    pKey = "jgvC0RrJREqY"
    endpoint = "https://s3g.data-infra.shopee.io"
    region = "default"
    s3 = boto3.client(
        "s3",
        aws_access_key_id=uKey,
        aws_secret_access_key=pKey,
        endpoint_url=endpoint,
        region_name=region,
    )
    print(f"Uploading {zip_path} to s3://{bucket}/{key}")
    s3.upload_file(zip_path, bucket, key)


def main(file1, file2, bucket="sg-sls-mart-ads-forecast-apb-model", s3_prefix="rep_model_onnx"):
    unix_time = int(time.time())
    version = f"{s3_prefix}_v{unix_time}"
    zip_filename = f"{version}.zip"

    zip_path = os.path.join("/tmp", os.path.basename(zip_filename))
    zip_files(file1, file2, zip_path)
    try:
        key = f"multiplier_model/{zip_filename}"
        upload_to_s3(zip_path, bucket, key)
        print(f"version: {key}")
        try:
            subprocess.run(["git", "tag", version], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing git command: {e}")
        except FileNotFoundError:
            print("Error: 'git' command not found. Make sure git is installed and in your PATH.")
    finally:
        os.remove(zip_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload ONNX model to S3")
    parser.add_argument("--onnx_file", help="onnx model path")
    parser.add_argument("--info_file", help="model config json path")
    args = parser.parse_args()
    main(args.onnx_file, args.info_file)
