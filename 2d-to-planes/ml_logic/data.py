from google.cloud import storage

def load_data_to_bucket(bloob):
    """
    - Load data from the bucket and a specific bloob
    - ???
    """

    storage_filename = "models/xgboost_model.joblib"
    local_filename = "model.joblib"

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(storage_filename)
    blob.download_to_filename(local_filename)
