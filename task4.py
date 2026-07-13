import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, from_unixtime, window, when, sum, count, avg
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.functions import when, sum as spark_sum

if __name__ == "__main__":

    spark = SparkSession.builder \
        .appName("task4") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    spark.conf.set("spark.sql.ansi.enabled", "false")

    # QUESTION 1 -------------------------------------------------------------------------------------------------------

    # Streaming source configuration
    smoke_host = os.getenv(
        "STREAMING_SERVER_SMOKE",
        "smoke-detection.stream-emulator.svc.cluster.local"
    )
    smoke_port = int(os.getenv("STREAMING_SERVER_SMOKE_PORT", "5551"))

    # Read stream from socket
    logsDF = spark.readStream \
        .format("socket") \
        .option("host", smoke_host) \
        .option("port", smoke_port) \
        .option("includeTimestamp", "true") \
        .load()

    # Split CSV columns
    splitCols = split(col("value"), ",")

    # Remove null / header rows (safer filtering)
    filteredDF = logsDF.filter(col("value").isNotNull()) \
                       .filter(~col("value").contains("UTC"))

    # Parse and cast required fields
    parsedDF = filteredDF.select(
        from_unixtime(splitCols.getItem(1).cast("long")).cast("timestamp").alias("event_time"),
        splitCols.getItem(2).cast(DoubleType()).alias("temperature"),
        splitCols.getItem(3).cast(DoubleType()).alias("humidity"),
        splitCols.getItem(4).cast(DoubleType()).alias("tvoc"),
        splitCols.getItem(5).cast(DoubleType()).alias("eco2"),
        splitCols.getItem(10).cast(DoubleType()).alias("pm25"),
        splitCols.getItem(15).cast(IntegerType()).alias("fire_alarm")
    )

    Print schema (REQUIRED for report)
    parsedDF.printSchema()

    # Write output to console (append mode, no truncation)
    query = parsedDF.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", "false") \
        .start()


    # Q2: Watermark + Window Aggregation -------------------------------------------------------------------------------------------
    
    # Apply watermark on event_time
    watermarkedDF = parsedDF.withWatermark("event_time", "5 seconds")
    
    # Window aggregation (REQUIRED to see watermark effect)
    windowedCountsDF = watermarkedDF.groupBy(
        window(col("event_time"), "10 seconds")
    ).count()
    
    # Write stream in APPEND mode (important for watermark behavior)
    query = windowedCountsDF.writeStream \
        .outputMode("append") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    # Q3a: Sliding Window Low-Humidity Count -------------------------------------------------------------------------------------------
    
    # # Apply watermark (same as Q2)
    watermarkedDF = parsedDF.withWatermark("event_time", "5 seconds")
    
    # Filter + count humidity < 50 using conditional aggregation
    lowHumidityDF = watermarkedDF.groupBy(
        window(col("event_time"), "60 seconds", "30 seconds")
    ).agg(
        sum(when(col("humidity") < 50, 1).otherwise(0)).alias("low_humidity")
    )
    
    # Write stream in UPDATE mode 
    query = lowHumidityDF.writeStream \
        .outputMode("update") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    # # -----------------------------------
    # # Q3b: Stateful Alarm-State Counting
    # # -----------------------------------
    
    # Stateful aggregation: running counts per fire_alarm
    alarmCountsDF = parsedDF.groupBy("fire_alarm").count()
    
    # Sort by descending count 
    sortedAlarmCountsDF = alarmCountsDF.orderBy(col("count").desc())
    
    # Write stream in COMPLETE mode 
    query = sortedAlarmCountsDF.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", "false") \
        .start()


    # Q4a: Aggregation by Alarm State -------------------------------------------------------------------------------------------------
    
    alarmAggDF = parsedDF.groupBy("fire_alarm").agg(
        avg("temperature").alias("avg_temp"),
        avg("pm25").alias("avg_pm25")
    )
    
    # Write in COMPLETE mode (REQUIRED)
    query = alarmAggDF.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    
    # Q4b: Triggered Aggregation of Risk Categories
    # Create high_risk column
    riskDF = parsedDF.withColumn(
        "high_risk",
        when((col("eco2") >= 415) & (col("humidity") >= 50), 1).otherwise(0)
    )
    
    # Group and count
    riskAggDF = riskDF.groupBy("high_risk").agg(
        count("*").alias("risk_count")
    )
    
    # Write with TRIGGER
    query = riskAggDF.writeStream \
        .outputMode("complete") \
        .trigger(processingTime="15 seconds") \
        .format("console") \
        .option("truncate", "false") \
        .start()

    
    # Q5 Window-Level Threshold Agreement--------------------------------------------------------------------------------------
    
    print("===== Q5: WINDOW-LEVEL THRESHOLD AGREEMENT =====")
    
    windowDuration = "60 seconds"
    slideDuration = "30 seconds"
    
    T = 20
    K = 1
    M = 1
    
    base_df = parsedDF \
        .filter(col("event_time").isNotNull()) \
        .withWatermark("event_time", "5 seconds")
    
    # Step 1: per-window counts
    window_counts_df = base_df \
        .groupBy(
            window(col("event_time"), windowDuration, slideDuration)
        ) \
        .agg(
            spark_sum(
                when(col("tvoc") >= T, 1).otherwise(0)
            ).alias("tvoc_alert_count"),
    
            spark_sum(
                when(
                    (col("eco2") >= 415) & (col("humidity") >= 50),
                    1
                ).otherwise(0)
            ).alias("risk_count")
        )
    
    # Step 2: create window-level binary labels
    window_labels_df = window_counts_df \
        .withColumn(
            "window_tvoc_alert",
            when(col("tvoc_alert_count") >= K, 1).otherwise(0)
        ) \
        .withColumn(
            "window_risk",
            when(col("risk_count") >= M, 1).otherwise(0)
        ) \
        .select(
            col("window.start").alias("window_start"),
            col("window.end").alias("window_end"),
            col("tvoc_alert_count"),
            col("risk_count"),
            col("window_tvoc_alert"),
            col("window_risk")
        )
    
    # Step 3: foreachBatch output
    def output_q5(batch_df, batch_id):
    
        print(f"===== Q5 BATCH {batch_id}: WINDOW LABELS =====")
        batch_df.orderBy("window_start").show(50, truncate=False)
    
        confusion_df = batch_df.groupBy(
            "window_tvoc_alert",
            "window_risk"
        ).count().orderBy(
            "window_tvoc_alert",
            "window_risk"
        )
    
        print(f"===== Q5 BATCH {batch_id}: CONFUSION MATRIX =====")
        confusion_df.show(20, truncate=False)
    
        rows = confusion_df.collect()
    
        TP = 0
        FP = 0
        FN = 0
    
        for row in rows:
            pred = row["window_tvoc_alert"]
            actual = row["window_risk"]
            c = row["count"]
    
            # FIX 3: use += (not =)
            if pred == 1 and actual == 1:
                TP += c
            elif pred == 1 and actual == 0:
                FP += c
            elif pred == 0 and actual == 1:
                FN += c
    
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    
        print(f"===== Q5 BATCH {batch_id}: PRECISION AND RECALL =====")
        print(f"TP = {TP}, FP = {FP}, FN = {FN}")
        print(f"Precision = {precision}")
        print(f"Recall = {recall}")
    
    
    # Step 4: start query (COMPLETE MODE required)
    query = window_labels_df.writeStream \
        .outputMode("complete") \
        .foreachBatch(output_q5) \
        .trigger(processingTime="10 seconds") \
        .start()

    
    # Q6 Fault Tolerance and Recovery --------------------------------------------------------------------------------------------
    
    print("===== Q6: FAULT TOLERANCE AND RECOVERY =====")
    
    checkpoint_path = f"s3a://bdp-student-ec25240/task4/checkpoints/q6_alarm_counts"
    
    # Reuse Q3b logic (stateful aggregation)
    alarm_counts_df = parsedDF.groupBy("fire_alarm").count()
    
    sorted_df = alarm_counts_df.orderBy(col("count").desc())
    
    #Start streaming query with checkpointing
    query = sorted_df.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", "false") \
        .option("checkpointLocation", checkpoint_path) \
        .start()
       

    # Keep streaming running
    query.awaitTermination()