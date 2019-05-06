from pyspark import SparkConf, SparkContext
from pyspark.sql.types import *
import pyspark
import numpy as np
from elephas.spark_model import SparkModel
from create_youtube_model import create_model

# ============
# SPARK SETUP
# ============

# NOTE: Run the below command when submitting this job as well:
# spark-submit --jars ecosystem/spark/spark-tensorflow-connector/target/spark-tensorflow-connector_2.11-1.10.0.jar train_youtube_elephas.py

conf = SparkConf().setAppName('Youtube-8M') \
                  .set("spark.jars",
                       "ecosystem/spark/spark-tensorflow-connector/target/spark-tensorflow-connector_2.11-1.10.0.jar")
sc = SparkContext(conf = conf)
spark = pyspark.sql.SparkSession(sc)

# =============
# READING DATA
# =============

# We use the Tensorflow-Spark connector (provided by Tensorflow)
# which allows us to read the tfrecord files into a Spark
# DataFrame.

# VIDEO-LEVEL
vid_train_df = spark.read.format("tfrecords").option("recordType", "Example").load('s3://cs205-youtube-data/yt8pm/v2/video/train*.tfrecord')
# vid_test_df = spark.read.format("tfrecords").option("recordType", "Example").load('s3://cs205-youtube-data/yt8pm/v2/video/test*.tfrecord')

# ==============
# PREPROCESSING
# ==============

# 1) take mean_rgb features
# 2) take top 10 classes
train_df = vid_train_df.select('mean_rgb', 'labels')
train_rdd = train_df.rdd
train_rdd = train_rdd.map(lambda x: (x[0], x[1]))
def convert_labels(labels):
    allowed=np.arange(10)
    one_hot=np.zeros(10)
    for l in labels:
        if l in allowed:
            one_hot[l]=1
    return one_hot
train_rdd = train_rdd.map(lambda x: (np.array(x[0]), convert_labels(x[1])))

# =========
# TRAINING
# =========

keras_model = create_model()
spark_model = SparkModel(keras_model, frequency='batch', mode='synchronous')
history = spark_model.fit(train_rdd, epochs=1, batch_size=32, verbose=2)

# =========
# TESTING
# =========
# score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
# print('Test accuracy:', score[1])
