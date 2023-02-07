from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import IsolationForest
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder.appName("IsolationForest").getOrCreate()

# Load the data into a PySpark DataFrame
data = spark.read.csv("data.csv", header=True, inferSchema=True)

# Encode the class labels as integers
stringIndexer = StringIndexer(inputCol="class_label", outputCol="class_index")
model = stringIndexer.fit(data)
data = model.transform(data)

# One-hot encode the class labels
encoder = OneHotEncoder(inputCol="class_index", outputCol="class_vec")
data = encoder.transform(data)

# Convert the data into a format that can be used by the Isolation Forest model
def vectorize_data(row):
    feature_vector = Vectors.dense(row[1:])
    return (row[0], feature_vector)

vectorized_data = data.rdd.map(vectorize_data)
vectorized_data = spark.createDataFrame(vectorized_data, ["class_label", "features"])

# Train an Isolation Forest model
isolation_forest = IsolationForest(contamination=0.1, seed=42)
model = isolation_forest.fit(vectorized_data)

# Use the model to make predictions
predictions = model.transform(vectorized_data)

# Add a new column to the DataFrame to indicate whether each sample is an outlier or not
predictions = predictions.withColumn("outlier", predictions["prediction"].cast("integer").alias("outlier"))
