<!--
{% comment %}
Licensed to the Apache Software Foundation (ASF) under one or more
contributor license agreements.  See the NOTICE file distributed with
this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0
(the "License"); you may not use this file except in compliance with
the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
{% endcomment %}
-->
# Deep Learning for Breast Cancer Mitosis Detection and Tumor Proliferation Score Prediction

## Overview
The [Tumor Proliferation Assessment Challenge 2016 (TUPAC16)](http://tupac.tue-image.nl/) is a "Grand Challenge" that was created for the [2016 Medical Image Computing and Computer Assisted Intervention (MICCAI 2016)](http://miccai2016.org/en/) conference.  In this challenge, the goal is to develop state-of-the-art algorithms for automatic prediction of tumor proliferation scores from whole-slide histopathology images of breast tumors.

## Background
Breast cancer is the leading cause of cancerous death in women in less-developed countries, and is the second leading cause of cancerous deaths in developed countries, accounting for 29% of all cancers in women within the U.S. [1]. Survival rates increase as early detection increases, giving incentive for pathologists and the medical world at large to develop improved methods for even earlier detection [2].  There are many forms of breast cancer including Ductal Carcinoma in Situ (DCIS), Invasive Ductal Carcinoma (IDC), Tubular Carcinoma of the Breast, Medullary Carcinoma of the Breast, Invasive Lobular Carcinoma, Inflammatory Breast Cancer and several others [3]. Within all of these forms of breast cancer, the rate in which breast cancer cells grow (proliferation), is a strong indicator of a patient’s prognosis. Although there are many means of determining the presence of breast cancer, tumor proliferation speed has been proven to help pathologists determine the best treatment for the patient. The most common technique for determining the proliferation speed is through mitotic count (mitotic index) estimates, in which a pathologist counts the dividing cell nuclei in hematoxylin and eosin (H&E) stained slide preparations to determine the number of mitotic bodies.  Given this, the pathologist produces a proliferation score of either 1, 2, or 3, ranging from better to worse prognosis [4]. Unfortunately, this approach is known to have reproducibility problems due to the variability in counting, as well as the difficulty in distinguishing between different grades.

References: <br />
[1] http://emedicine.medscape.com/article/1947145-overview#a3 <br />
[2] http://emedicine.medscape.com/article/1947145-overview#a7 <br />
[3] http://emedicine.medscape.com/article/1954658-overview <br />
[4] http://emedicine.medscape.com/article/1947145-workup#c12 <br />

## Goal & Approach

### Mitosis Detection
At a high level, as shown in Figure 1, our approach begins by preprocessing a dataset of regions of tissue from whole slide images (WSIs) of breast tumors into a dataset of mitotic and non-mitotic patches. We then train a convolutional neural network (CNN) model to predict the presence of a mitotic figure in a given patch. Given an initial trained model, we preprocess the raw dataset again with model-based false-positive (FP) oversampling to generate a more difficult training dataset. We then train a new model on this second dataset. To make predictions on a new image region, we apply the model to the image in a sliding window fashion with noise marginalization, yielding a prediction at each location. A clustering algorithm is then used to smooth the potentially noisy set of predictions into a set of final predictions for mitosis locations.

### Tumor Proliferation Score Prediction (previous approach)
In an effort to automate the process of classification, this project aims to develop a large-scale deep learning approach for predicting tumor scores directly from the pixels of whole-slide histopathology images (WSI).  Our proposed approach is based on a recent research paper from Stanford [1].  Starting with 500 extremely high-resolution tumor slide images [2] with accompanying score labels, we aim to make use of Apache Spark in a preprocessing step to cut and filter the images into smaller square samples, generating 4.7 million samples for a total of ~7TB of data [3].  We then utilize TensorFlow and Keras to train a deep convolutional neural network on these samples, making use of transfer learning by fine-tuning a modified ResNet50 model [4].  Our model takes as input the pixel values of the individual samples, and is trained to predict the correct tumor score classification for each one.  We also explore an alternative approach of first training a mitosis detection model [5] on an auxiliary mitosis dataset, and then applying it to the WSIs, based on an approach from Paeng et al. [6].  Ultimately, we aim to develop a model that is sufficiently stronger than existing approaches for the task of breast cancer tumor proliferation score classification.

References: <br />
[1] https://web.stanford.edu/group/rubinlab/pubs/2243353.pdf <br />
[2] http://tupac.tue-image.nl/node/3 <br />
[3] [`preprocess.py`](preprocess.py), [`breastcancer/preprocessing.py`](breastcancer/preprocessing.py) <br />
[4] [`MachineLearning-Keras-ResNet50.ipynb`](MachineLearning-Keras-ResNet50.ipynb) <br />
[5] [`preprocess_mitoses.py`](preprocess_mitoses.py), [`train_mitoses.py`](train_mitoses.py) <br />
[6] https://arxiv.org/abs/1612.07180

![Approach](approach.jpg)

## Steps for Mitosis Detection
### Packages
- `pip3 install -U numpy keras tensorflow pillow pandas dask scikit-learn pytest`

### Help
- all script files have a `--help` option that will output the command-line options

### Raw data (on a shared filesystem on the ram machines):
- images: `data/mitoses/mitoses_train_image_data` (the preprocessing code expects, by default, the `data/mitoses/mitoses_train_image_data` path to be available from the base directory of the project)
- labels: `data/mitoses/mitoses_train_ground_truth`, (same as above for preprocessing code)

### Preprocessing
- basic: `python3 preprocess_mitoses.py --save_path=data/mitoses/patches --rotations_train=0 --translations_train=0 --p_train=0.0001 --p_val=0.0001 --seed 0`

### Training
- basic logreg: `python3 train_mitoses.py --patches_path=data/mitoses/patches --model=logreg`
- basic logreg on GPU 0 of a GPU server: `CUDA_VISIBLE_DEVICES=0 python3 train_mitoses.py --patches_path=data/mitoses/patches --model=logreg`

### Evaluation (optional)
- eval above model: `python3 eval_mitoses.py --model_name=logreg --model_path=PATH_TO_EXPERIMENT_FOLDER_FROM_TRAINING/checkpoints/BEST_MODEL_IN_THIS_FOLDER.hdf5 --patches_path=data/mitoses/patches/val`
- eval above model on GPU 0 of a GPU server: `CUDA_VISIBLE_DEVICES=0 python3 eval_mitoses.py --model_name=logreg --model_path=PATH_TO_EXPERIMENT_FOLDER_FROM_TRAINING/checkpoints/BEST_MODEL_IN_THIS_FOLDER.hdf5 --patches_path=data/mitoses/patches/val`

### Hyperparameter Tuning
- basic logreg: `python3 hyperparam_tune_mitoses.py --patches_path=data/mitoses/patches --models logreg --log_interval 1000`
- basic logreg on GPU 0 of a GPU server: `CUDA_VISIBLE_DEVICES=0 python3 hyperparam_tune_mitoses.py --patches_path=data/mitoses/patches --models logreg --log_interval 1000`

### Testing
- `pytest file.py`

### Prediction template
```
import numpy as np
from tf.keras.models import load_model

from train_mitoses import normalize

model_file = ...  # hdf5 file, no sigmoid at the end
model_name = ....  # "vgg", "resnet", or "logreg" currently
threshold = ...  # scalar value
patch_batch = ...  # numpy array of shape (N, 64, 64, 3) containing N images

# load the model and add the sigmoid layer if we want to use easier-to-interpret threshold values
base_model = load_model(model_file, compile=False)
probs = keras.layers.Activation('sigmoid', name="sigmoid")(base_model.output)
model = keras.models.Model(inputs=base_model.input, outputs=probs)

# prediction
norm_patch_batch = normalize((np.array(patch_batch) / 255).astype(np.float32), model_name)  # shape (N, 64, 64, 3)
out_batch = model.predict_on_batch(norm_patch_batch)  # shape (N, 1) with probs
pred = out_batch > threshold  # shape (N, 1) with binary predictions
```

## Setup for Tumor Proliferation Score Prediction (*All nodes* unless other specified):
* System Packages:
  * `openslide`
* Python packages:
  * Basics
    * `pip3 install -U matplotlib numpy pandas scipy jupyter ipython scikit-learn scikit-image openslide-python`
  * TensorFlow (only on driver):
    * `pip3 install tensorflow-gpu` (or `pip3 install tensorflow` for CPU-only)
  * Keras (bleeding-edge; only on driver):
    * `pip3 install git+https://github.com/fchollet/keras.git`
* Spark 2.x (ideally bleeding-edge)
* Add the following to the `data` folder (same location on *all* nodes):
  * `training_image_data` folder with the training slides.
  * `testing_image_data` folder with the testing slides.
  * `training_ground_truth.csv` file containing the tumor & molecular scores for each slide.
  * `mitoses` folder with the following from the mitosis detection auxiliary dataset:
    * `mitoses_test_image_data` folder with the folders of testing images
    * `mitoses_train_image_data` folder with the folders of training images
    * `mitoses_train_ground_truth` folder with the folders of training csv files
* Layout:
  ```
  - MachineLearning-Keras-ResNet50.ipynb
  - breastcancer/
    - preprocessing.py
    - visualization.py
  - ...
  - data/
    - mitoses
      - mitoses_test_image_data
        - 01
          - 01.tif
        - 02
          - 01.tif
        ...
      - mitoses_train_ground_truth
        - 01
          - 01.csv
          - 02.csv
          ...
        - 02
          - 01.csv
          - 02.csv
          ...
        ...
      - mitoses_train_image_data
        - 01
          - 01.tif
          - 02.tif
          ...
        - 02
          - 01.tif
          - 02.tif
          ...
        ...
    - training_ground_truth.csv
    - training_image_data
      - TUPAC-TR-001.svs
      - TUPAC-TR-002.svs
      - ...
    - testing_image_data
      - TUPAC-TE-001.svs
      - TUPAC-TE-002.svs
      - ...
  - preprocess.py
  - preprocess_mitoses.py
  - train_mitoses.py
  ```

* Adjust the Spark settings in `$SPARK_HOME/conf/spark-defaults.conf` using the following examples, depending on the job being executed:
  * All jobs:
    ```
    # Use most of the driver memory.
    spark.driver.memory 70g
    # Remove the max result size constraint.
    spark.driver.maxResultSize 0
    # Increase the message size.
    spark.rpc.message.maxSize 128
    # Extend the network timeout threshold.
    spark.network.timeout 1000s
    # Setup some extra Java options for performance.
    spark.driver.extraJavaOptions -server -Xmn12G
    spark.executor.extraJavaOptions -server -Xmn12G
    # Setup local directories on separate disks for intermediate read/write performance, if running
    # on Spark Standalone clusters.
    spark.local.dirs /disk2/local,/disk3/local,/disk4/local,/disk5/local,/disk6/local,/disk7/local,/disk8/local,/disk9/local,/disk10/local,/disk11/local,/disk12/local
    ```

  * Preprocessing:
    ```
    # Save 1/2 executor memory for Python processes
    spark.executor.memory 50g
    ```

* To execute the WSI preprocessing script, use `spark-submit` as follows (could also use Yarn in client mode with `--master yarn --deploy-mode client`):
  ```
  PYSPARK_PYTHON=python3 spark-submit --master spark://MASTER_URL:7077 preprocess.py
  ```

* To execute the mitoses preprocessing script, use the following:
  ```
  python3 preprocess_mitoses.py --help
  ```

* To execute the mitoses training script, use the following:
  ```
  python3 training_mitoses.py --help
  ```

* To use the Jupyter notebooks, start up Jupyter like normal with `jupyter notebook` and run the desired notebook.

## Create a Histopath slide “lab” to view the slides (just driver):
  - `git clone https://github.com/openslide/openslide-python.git`
  - Host locally:
    - `python3 path/to/openslide-python/examples/deepzoom/deepzoom_multiserver.py -Q 100 path/to/data/`
  - Host on server:
    - `python3 path/to/openslide-python/examples/deepzoom/deepzoom_multiserver.py -Q 100 -l HOSTING_URL_HERE path/to/data/`
    - Open local browser to `HOSTING_URL_HERE:5000`.
