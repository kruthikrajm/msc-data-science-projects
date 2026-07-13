from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct
from pyspark.sql.functions import col, to_date, month, dayofweek, hour,to_timestamp
from pyspark.sql.functions import count, countDistinct, sum, col, when, avg
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("task1") \
        .getOrCreate()
    
    user_id = "ec25240"
    
    # Question 1. Load Dataset --------------------------------------------------------------------------------------------
    df = spark.read.csv(
        "s3a://module-big-data-processing/Expedia/hotels.csv",
        header=True,
        inferSchema=True
    )

    # 2. Printing the Schema
    print("===== DATAFRAME SCHEMA =====")
    df.printSchema()

    
    # 3. Printing Total number of rows
    total_rows = df.count()
    print("===== TOTAL ROWS (IMPRESSIONS) =====")
    print(total_rows)

    # 4. Distinct search IDs
    distinct_searches = df.select(countDistinct("srch_id")).collect()[0][0]
    print("===== DISTINCT SEARCHES (srchid) =====")
    print(distinct_searches)

    # 5. Cache DataFrame
    df.cache()
    df.count()

    # SAVE Question-1 OUTPUT TO CSV

    from pyspark.sql import Row

    # Creating a DataFrame for total impression and distinct seaches
    q1_result_df = spark.createDataFrame([
        Row(metric="total_impressions", value=total_rows),
        Row(metric="distinct_searches", value=distinct_searches)
    ])

    # Writing the ouput to CSV
    q1_result_df.coalesce(1).write.csv(
        "s3a://bdp-student-ec25240/task1/q1/",
        header=True,
        mode="overwrite"
    )

    
    # Question 2: TIME FEATURE ENGINEERING--------------------------------------------------------------------------------

    # 1. Convert datetime to timestamp
    df_timestamp = df.withColumn(
        "date_time",
        to_timestamp(col("date_time"))
    )

    df_timestamp = df_timestamp.withColumn("date", to_date(col("date_time"))) \
           .withColumn("month", month(col("date_time"))) \
           .withColumn("dow", dayofweek(col("date_time"))) \
           .withColumn("hour", hour(col("date_time")))


    # Show sample rows
    print("===== SAMPLE WITH TIME FEATURES =====")
    df_timestamp.select("srch_id", "date_time", "date", "month", "dow", "hour") \
      .show(10, truncate=False)

    
    # SAVE ALL ROWS TO CSV
    df_timestamp.select("srch_id", "date_time", "date", "month", "dow", "hour") \
        .limit(10) \
        .coalesce(1) \
        .write.csv(
            "s3a://bdp-student-ec25240/task1/q2_sample_10_rows/",
            header=True,
            mode="overwrite"
    )
    
    # Question 3a: SEARCH ACTIVITY OVER TIME------------------------------------------------------------------------------------
    
    activity_df = df_timestamp.groupBy("date") \
        .agg(
            count("*").alias("total_impressions"),
            countDistinct("srch_id").alias("total_searches"),
            sum("click_bool").alias("total_clicks"),
            sum("booking_bool").alias("total_bookings")
        )


    # Q3b: DERIVED METRICS

    activity_df = activity_df.withColumn(
            "CTR",
            col("total_clicks") / col("total_impressions")
        ) \
        .withColumn(
            "conversion_rate",
            col("total_bookings") / col("total_impressions")
        ) \
        .orderBy("date")

    # Showing 10 rows 
    print("===== FINAL AGGREGATED TABLE (10 ROWS) =====")
    activity_df.show(10, truncate=False)

    activity_df.coalesce(1).write.csv(
        "s3a://bdp-student-ec25240/task1/q3a_activity/",
        header=True,
        mode="overwrite"
    )


    # Question 4: TIME-OF-DAY BEHAVIOUR ------------------------------------------------------------------------------------------

    q4_df = df_timestamp.groupBy("hour", "srch_saturday_night_bool") \
        .agg(
            count("*").alias("total_impressions"),
            sum("click_bool").alias("total_clicks")
        ) \
        .withColumn(
            "CTR",
            col("total_clicks") / col("total_impressions")
        ) \
        .orderBy("hour", "srch_saturday_night_bool")

    
    # Showing results
    print("===== TIME-OF-DAY BEHAVIOUR =====")
    q4_df.show(24, truncate=False)

    q4_df.coalesce(1).write.csv(
        "s3a://bdp-student-ec25240/task1/q4_time_of_day/",
         header=True,
         mode="overwrite"
    )


    # Question 5: POSITION BIAS ANALYSIS ----------------------------------------------------------------------------------------

    # 1. Create star_band column
    df_q5 = df_timestamp.withColumn(
        "star_band",
        when(col("prop_starrating") <= 2, "0-2")
        .when(col("prop_starrating") == 3, "3")
        .when(col("prop_starrating") == 4, "4")
        .when(col("prop_starrating") == 5, "5")
    )

    # 2. Grouping and aggregate
    q5_df = df_q5.groupBy("position", "star_band") \
        .agg(
            sum("click_bool").alias("total_clicks"),
            count("*").alias("total_impressions"),
            sum("booking_bool").alias("total_bookings")
        ) \
        .withColumn(
            "CTR",
            col("total_clicks") / col("total_impressions")
        ) \
        .withColumn(
            "conversion_rate",
            col("total_bookings") / col("total_impressions")
        ) \
        .orderBy("position", "star_band")

    
    # Showing results 
    print("===== POSITION BIAS ANALYSIS =====")
    q5_df.show(20, truncate=False)

    q5_df.coalesce(1).write.csv(
        "s3a://bdp-student-ec25240/task1/q5/",
        header=True,
        mode="overwrite"
    )

    
    # Question 6: MONTHLY DESTINATION ANALYSIS -------------------------------------------------------------------------------------

    
    # 1. Aggregate metrics
    q6_df = df_timestamp.groupBy("month", "srch_destination_id") \
        .agg(
            count("*").alias("total_impressions"),
            sum("booking_bool").alias("total_bookings")
        ) \
        .withColumn(
            "conversion_rate",
            col("total_bookings") / col("total_impressions")
        )

    
    # 2. Top 10 by total bookings per month
    window_bookings = Window.partitionBy("month").orderBy(col("total_bookings").desc())

    top10_bookings = q6_df.withColumn(
        "rank",
        row_number().over(window_bookings)
    ).filter(col("rank") <= 10)


    # 3. Top 10 by conversion rate per month
    window_conversion = Window.partitionBy("month").orderBy(col("conversion_rate").desc(), col("total_bookings").desc(),col("total_impressions").desc(),
        col("srch_destination_id").asc())

    top10_conversion = q6_df.withColumn(
        "rank",
        row_number().over(window_conversion)
    ).filter(col("rank") <= 10)

    
    # Showing results
    print("===== TOP 10 DESTINATIONS BY BOOKINGS =====")
    top10_bookings.show(20, truncate=False)

    print("===== TOP 10 DESTINATIONS BY CONVERSION RATE =====")
    top10_conversion.show(20, truncate=False)
    
    top10_bookings.coalesce(1).write.csv(
        "s3a://bdp-student-ec25240/task1/q6/",
        header=True,
        mode="overwrite"
    )

    top10_conversion.coalesce(1).write.csv(
        "s3a://bdp-student-ec25240/task1/q6_coverstion/",
        header=True,
        mode="overwrite"
    )


    
    # Question 7: DETECTING UNUSUAL ACTIVITY ------------------------------------------------------------------------------------

    daily_search_df = df_timestamp.groupBy("date") \
        .agg(
            countDistinct("srch_id").alias("total_searches")
        ) \
        .orderBy("date")

    # 2. Compute mean daily searches
    mean_value = daily_search_df.select(avg("total_searches")).collect()[0][0]

    print("===== MEAN DAILY SEARCH COUNT =====")
    print(mean_value)

    
    # 3. Filter unusual days
    unusual_days_df = daily_search_df.filter(
        col("total_searches") > mean_value
    )

    
    # Showing results 
    print("===== UNUSUAL SEARCH ACTIVITY DAYS =====")
    unusual_days_df.show(20, truncate=False)

    daily_search_df.coalesce(1).write.csv(
        "s3a://bdp-student-ec25240/task1/q7/",
        header=True,
        mode="overwrite"
    )

    unusual_days_df.coalesce(1).write.csv(
        "s3a://bdp-student-ec25240/task1/q7_unusual_days/",
        header=True,
        mode="overwrite"
    )

    
    spark.stop()