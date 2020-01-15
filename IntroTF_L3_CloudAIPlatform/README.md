# Lesson 3: Cloud AI Platform

## Task.py
* The entry point to the code
* Start **Job Level** details (made it by command line arguments)
    * command line arguments
    * how long to run
    * where to write outputs
    * hyper parameter tuning
* Invoke model.py


## model.py
* focus on core ML tasks
* fetch data
* define features
* configure service signature
* train and evaluate loops

## Package the TF model as Python package
```
    taxifare/
    taxifare/PKG-INFO
    taxifare/setup.cfg
    taxifare/setup.py
    taxifare/trainer/
    taxifare/trainer/__init__.py    # python module needs to contain an __init__.py in every folder
    taxifare/trainer/task.py
    taxifare/trainer/model.py
```
* Use **python -m** to check all imports are in the good shape or not

## Gcloud
* Use **gcloud** command to submit training job
```
    # Local TRain
    gcloud ml-engine local train \
        --module-name = trainer.task \
        --package-path = .../taxifare/trainer \
        ...

    # On Gcloud
    gcloud ml-engine jobs submit training $JOBNAME \
        --region = $REGION \
        --module-name = trainer.task \
        --job-dir = $OUT_DIR
        --scale-tier = BASIC
        ...
```
* --package-path: specify where the codes are located
* --module-name: specify which of the files in the package to execute
* --scale-tier: specify what kind of hardware you want to use
    * BASIC: one machine
    * STANDAED: small cluster
    * BASIC_GPU: single GPU
    * BASIC_TPU: single TPU
    * custom machine type
* Use **single-region** bucket for ML to get better performance
    * default is multi-region -> it is better for web serving


## Monitor model
```
    # Get details of current stat of job
    gcloud ml-engine jobs describe job_name

    # Get latest logs from job
    gcloud ml-engine jobs stream-job job_name

    # Filter jobs based on creation time or name
    gcloud ml-engine jobs list --filter='createTime>...'
    gcloud ml-engine jobs list --filter='jobID:census*' --limit=3
```

## Deploy model to GCP
```
    MODEL_NAME = 'taxifare'
    MODEL_VERSION = 'v1'
    MODEL_LOCATION = 'gs://....'


    gcloud ml-engine models create ${MODEL_NAME} --regions $REGION
    gcloud ml-engine versions create ${MODEL_VERSION} --model ${MODEL_NAME} --origin ${MODEL_LOCATION} --runtime-version 1.4
```

## Client code to make REST calls from deployed model
```
    credentials = GoogleCredentials.get_application_default()
    api = discovery.build('m1', 'v1', credentials=credentials)
    request_data = [
        {
            ...
        }
    ]

    parent = 'projects/...'
    response = api.projects().predict(body={...})
    name = parent.execute()
```

LAB: *training-data-analyst > courses > machine_learning > deepdive > 03_tensorflow > labs*and open *e_ai_platform.ipynb*.