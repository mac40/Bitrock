from pyspark.sql import SparkSession

from pyspark.sql.functions import (
    from_unixtime,
    to_timestamp,
    when,
    date_format,
    sum,
    col,
    desc,
    size,
    collect_set,
    max,
)
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    LongType,
)

from pyspark.sql.window import Window


def main() -> None:
    # Create a Spark session
    spark = SparkSession.builder.appName("PlacementStatistics").getOrCreate()

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

    # Read the DataFrame from the Parquet file, using the specified schema
    df = spark.read.schema(schema).parquet("data/")

    # Convert 'Timestamp' to a timestamp type
    df = df.withColumn("Timestamp", to_timestamp(from_unixtime(df["Timestamp"] / 1000)))

    # Create views and clicks columns
    df = df.withColumn("Views", when(df["Event_type"] == 0, 1).otherwise(0))
    df = df.withColumn("Clicks", when(df["Event_type"] == 1, 1).otherwise(0))

    # Define windows for hourly and daily aggregation
    hourly_window = Window.partitionBy(
        "Placement_id", date_format("Timestamp", "yyyy-MM-dd HH")
    ).orderBy("Timestamp")
    daily_window = Window.partitionBy(
        "Placement_id", date_format("Timestamp", "yyyy-MM-dd")
    ).orderBy("Timestamp")

    # Compute hourly and daily statistics
    hourly_stats = (
        df.withColumn("Hourly_Views", sum("Views").over(hourly_window))
        .withColumn("Hourly_Clicks", sum("Clicks").over(hourly_window))
        .withColumn(
            "Distinct_Users", size(collect_set("User_id").over(hourly_window))
        )
        .groupBy("Placement_id")
        .agg(
            max("Hourly_Views").alias("Max_Hourly_Views"),
            max("Hourly_Clicks").alias("Max_Hourly_Clicks"),
            max("Distinct_Users").alias("Max_Distinct_Users"),
        )
        .select(
            "Placement_id",
            "Max_Hourly_Views",
            "Max_Hourly_Clicks",
            "Max_Distinct_Users",
        )
    )

    daily_stats = (
        df.withColumn("Daily_Views", sum("Views").over(daily_window))
        .withColumn("Daily_Clicks", sum("Clicks").over(daily_window))
        .withColumn(
            "Distinct_Users", size(collect_set("User_id").over(daily_window))
        )
        .groupBy("Placement_id")
        .agg(
            max("Daily_Views").alias("Max_Daily_Views"),
            max("Daily_Clicks").alias("Max_Daily_Clicks"),
            max("Distinct_Users").alias("Max_Distinct_Users"),
        )
        .select(
            "Placement_id",
            "Max_Daily_Views",
            "Max_Daily_Clicks",
            "Max_Distinct_Users",
        )
    )

    # Show the results
    print("Hourly Statistics:")
    hourly_stats.show()

    print("Daily Statistics:")
    daily_stats.show()

    # Stop the Spark session
    spark.stop()


if __name__ == "__main__":
    main()
