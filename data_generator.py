from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, LongType, IntegerType, StringType

# Create a Spark session
spark = SparkSession.builder.appName("DataGeneration").getOrCreate()

# Generate sample data using pandas
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# Generate random timestamps within a specific range using pandas
start_timestamp = (
    int((datetime.utcnow() - timedelta(days=30)).timestamp()) * 1000
)  # in milliseconds
end_timestamp = int(datetime.utcnow().timestamp()) * 1000  # in milliseconds
timestamps = np.random.randint(start_timestamp, end_timestamp, size=1000000)

# Generate other random data using pandas
event_types = np.random.choice([0, 1], size=1000000)
banner_ids = ["Banner" + str(i) for i in range(1, 101)]
placement_ids = ["Placement" + str(i) for i in range(1, 101)]
page_ids = ["Page" + str(i) for i in range(1, 101)]
user_ids = ["User" + str(i) for i in range(1, 101)]

# Create a pandas DataFrame
pandas_df = pd.DataFrame(
    {
        "Timestamp": timestamps,
        "Event_type": event_types,
        "Banner_id": np.random.choice(banner_ids, size=1000000),
        "Placement_id": np.random.choice(placement_ids, size=1000000),
        "Page_id": np.random.choice(page_ids, size=1000000),
        "User_id": np.random.choice(user_ids, size=1000000),
    }
)

# Convert pandas DataFrame to PySpark DataFrame
schema = StructType(
    [
        StructField("Timestamp", LongType(), True),
        StructField("Event_type", IntegerType(), True),
        StructField("Banner_id", StringType(), True),
        StructField("Placement_id", StringType(), True),
        StructField("Page_id", StringType(), True),
        StructField("User_id", StringType(), True),
    ]
)

# Use SparkSession to create a PySpark DataFrame
df = spark.createDataFrame(pandas_df, schema=schema)

# Show the PySpark DataFrame
df.write.mode("overwrite").parquet('data/')

# Stop the Spark session
spark.stop()
