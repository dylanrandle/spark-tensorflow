from pyspark import SparkConf, SparkContext
from pyspark.sql.types import *
import pyspark
import numpy as np
from elephas.spark_model import SparkModel
from create_and_train_biLSTM_Youtube import create_model

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

path = lambda m: "yt8pm_100th_shard/v2/video/{}*.tfrecord".format(m)
train_df = spark.read.format("tfrecords").option("recordType", "Example").load(path('train'))
val_df = spark.read.format("tfrecords").option("recordType", "Example").load(path('validate'))
train_df = spark.read.format("tfrecords").option("recordType", "Example").load(path('test'))

# ==============
# PREPROCESSING
# ==============

# 1) take audio and rgb features
# 2) use only top 20 classes
train_df = train_df.select('mean_rgb', 'labels')
train_rdd = train_df.rdd
train_rdd = train_rdd.map(lambda x: (x[0], x[1]))
def convert_labels(labels):
    allowed=np.arange(20)
    one_hot=np.zeros(20)
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
history = spark_model.fit(train_rdd, epochs=10, batch_size=32, verbose=0)

# =========
# TESTING
# =========
# score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
# print('Test accuracy:', score[1])
