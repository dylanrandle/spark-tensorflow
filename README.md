# CS205 Spring 2019 Final Project
Youtube video classification at scale using distributed computing and bi-LSTM.

We propose Youtube video classification at scale, leveraging a dataset of over 8 million videos with high throughput computing and large data handling system design.  The project requires training on terabytes of video and audio files to then predict a video type from several thousand labels.  Our solution uses two bidirectional-LSTM networks, one for audio and one for video, trained on a spark based distributed file system enabled by Elephas: Distributed Deep Learning with Keras & Spark.  The infrastructure used consists of a custom cluster of p2xlarge instances on AWS using amazon machine images to spin-up compatible nodes.  We found this approach produced reliable systemic performance and effective classification with our final model.


## Setup and Installation:

Below we describe the various steps required to run the code and reproduce execution.

### Getting Data:

Please find the YouTube-8M dataset [here](https://research.google.com/youtube8m/download.html). Follow the instructions to transfer the data to an S3 bucket.

### Creating a Machine Image:

- Launch an Amazon EC2 `m4.xlarge` instance with **Amazon Linux 2018.03.0** as the operating system.
- Install Python 3.6 with `yum install python36`
- Install a large collection of dependencies with `python3 -m pip install elephas`
- **Critical:** Uninstall the `pyspark` package installed by the previous command with `python3 -m pip uninstall pyspark`
- Go to the AWS console, select this instance, and create an image by selecting Action > Image > Create Image

### Launching the EMR Cluster with an AMI:

- Go to EMR, select Create Cluster and then go to Advanced Options
- Using `emr 5.23` dependencies, select:
  - Spark 2.4.0 
  - Hadoop 2.8.5
  - Ganglia 3.7.2
  - Zeppelin 0.8.1
- Size the cluster as your wish, but ensure the Master's EBS Root Volume is >= the AMI's EBS Root Volume (created above)
- On the last page, under images, select the previously created machine image

### Setting up the Cluster:

- `ssh` into the master node
- You may need to install Git with `yum install git`
- Follow the directions in the notebook: `Tensorflow-spark-connector.ipynb` to install the Tensorflow-Spark connector
  - In essence this involves a) installing Apache Maven and b) using it to install Tensorflow-Spark connector
- (Optional): follow the instructions [here](https://cloudkul.com/blog/mounting-s3-bucket-linux-ec2-instance/) to mount the S3 bucket on your local machine, then use `hadoop distcp` to move the files onto HDFS. The code reads directly from S3 so you would need to change the path if you do this step. We believe it may provide some performance benefits, but elected to stick with S3 for simplicity.
- Add to your `~/.bashrc`: 
```
export PATH=/usr/lib/spark:$PATH
export PYSPARK_PYTHON=/usr/bin/python3
```
- Reload it with `source ~/.bashrc`
- Clone this repo to the master or copy `train_youtube_elephas.py` and `create_youtube_model.py` to the master
- Submit the Spark job with: `spark-submit --jars ecosystem/spark/spark-tensorflow-connector/target/spark-tensorflow-connector_2.11-1.10.0.jar train_youtube_elephas.py`

Watch as Spark distributes the dataset and performs model training!

### Memory Issues

We had to tune the maximum memory allotted to the various processes. If you run into any trouble with memory, consider adjusting the flags `--driver-memory` to change the maximum memory available to the driver script, `--executor-memory` for the executors, and finally `--conf spark.driver.maxResultSize=SIZE` where `SIZE` is the maximum expected serialized result your `model.train` method returns.

### Monitoring

To easily monitor the Spark application, including tracking the runtime of each Job, enable port forwarding with:
`ssh -i /path/to/key -4 -L 3000:MASTER-DNS:4040 hadoop@MASTER-DNS`
where `MASTER-DNS` is the DNS found on the EMR cluster page.

In our experiments, we set `--num-executors` to the number of nodes in the cluster and `--executor-cores` to the number of vCPUs per node.
