from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, when
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.functions import col
from pyspark.sql.functions import from_unixtime, to_timestamp, date_format, year, month, when, avg, count,lag, explode, split
from pyspark.sql import Row
from pyspark.sql.functions import  round
from pyspark.sql.window import Window


if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Task2") \
        .getOrCreate()

    userId = "ec25240"

    # Question 1 -------------------------------------------------------------------------------------------------------------
    
    # 1. Load Data from S3
    
    # ratings_path = "s3a://bdp-student-ec25240/sample_data/task2/Books_rating_sample.csv"
    # books_path = "s3a://bdp-student-ec25240/sample_data/task2/books_data.csv"
    
    ratings_path = "s3a://module-big-data-processing/AmazonBooks/Books_rating.csv"
    books_path = "s3a://module-big-data-processing/AmazonBooks/books_data.csv"

    ratings_df = spark.read.csv(ratings_path, header=True, inferSchema=True)
    books_df = spark.read.csv(books_path, header=True, inferSchema=True)

    # 2. Data Cleaning

    # Replacing null categories with "Other"
    books_df = books_df.withColumn(
        "categories",
        when(col("categories").isNull(), "Other").otherwise(col("categories"))
    )

    # Remove invalid review_time rows (e.g., -1)
    ratings_df = ratings_df.filter(col("review_time") != -1)

    # Converting data types 
    ratings_df = ratings_df.withColumn("review_score", col("review_score").cast(FloatType()))
    ratings_df = ratings_df.withColumn("review_time", col("review_time").cast(IntegerType()))

    # 3. Printing Schema 
    print("Ratings Schema:")
    ratings_df.printSchema()

    print("Books Schema:")
    books_df.printSchema()

    # 4. Compute Required Metrics
    total_reviews = ratings_df.count()

    distinct_users = ratings_df.select("user_id").distinct().count()

    distinct_books = ratings_df.select("id").distinct().count()

    
    # 5. Print Results 
    print("======================================")
    print("Total Reviews:", total_reviews)
    print("Distinct Users:", distinct_users)
    print("Distinct Books:", distinct_books)
    print("=======================================")

  
    
    # Create DataFrame for Q1 results
    q1_results_df = spark.createDataFrame([
        Row(metric="Total Reviews", value=total_reviews),
        Row(metric="Distinct Users", value=distinct_users),
        Row(metric="Distinct Books", value=distinct_books)
    ])
    
    # Write to S3 as CSV
    q1_results_df.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task2/q1_results/")

    
    # Question 2: Time Conversion and Sorting -----------------------------------------------------------------------------------------
    
    # Convert review_time → timestamp
    ratings_df = ratings_df.withColumn(
        "timestamp",
        from_unixtime(col("review_time"))
    )
    
    # Create formatted date (YYYY-MM-DD)
    ratings_df = ratings_df.withColumn(
        "formatted_date",
        date_format(col("timestamp"), "yyyy-MM-dd")
    )
    
    # Extract Year and Month
    ratings_df = ratings_df.withColumn("year", year(col("timestamp")))
    ratings_df = ratings_df.withColumn("month", month(col("timestamp")))
    
    # Categorize time of rating
    ratings_df = ratings_df.withColumn(
        "year_period",
        when(col("month") <= 6, "Early Year").otherwise("Late Year")
    )
    
    # Sort dataset by date
    sorted_df = ratings_df.orderBy(col("timestamp"))
    
    # Show 10 sample rows
   
    print("=====10 sorted samples rows================")
    sorted_df.select(
        "id",
        "user_id",
        "review_score",
        "review_time",
        "timestamp",
        "formatted_date",
        "year",
        "month",
        "year_period"
    ).show(10, truncate=False)

    sorted_df.select(
        "id",
        "user_id",
        "review_score",
        "review_time",
        "timestamp",
        "formatted_date",
        "year",
        "month",
        "year_period"
    ).limit(10) \
     .coalesce(1) \
     .write \
     .mode("overwrite") \
     .option("header", True) \
     .csv("s3a://bdp-student-ec25240/task2/q2_results/")


    # Question 3: Rating Distribution and User Behaviour -----------------------------------------------------------------------------

    
    # 1. Create Rating Bands
    ratings_df = ratings_df.withColumn(
        "rating_band",
        when(col("review_score") <= 2, "Low")
        .when((col("review_score") >= 3) & (col("review_score") <= 4), "Medium")
        .when(col("review_score") == 5, "High")
        .otherwise("Other")
    )
    
    # 2. Total number of reviews 
    total_reviews = ratings_df.count()
    
    # 3. Grouping by rating band
    band_stats = ratings_df.groupBy("rating_band").agg(
        count("*").alias("total_reviews"),
        countDistinct("user_id").alias("distinct_users"),
        countDistinct("id").alias("distinct_books")
    )
    
    
    # 4. Percentage of total reviews
    band_stats = band_stats.withColumn(
        "percentage_reviews",
        (col("total_reviews") / total_reviews) * 100
    ).withColumn(
        "avg_reviews_per_user",
        (col("total_reviews") / col("distinct_users"))
    ).orderBy(col("total_reviews").desc())
    
    # -------------------------------
    # 5. Average reviews per user in each band
    # -------------------------------
    # user_band = ratings_df.groupBy("rating_band", "user_id").agg(
    #     count("*").alias("user_review_count")
    # )
    
    # avg_user_reviews = user_band.groupBy("rating_band").agg(
    #     avg("user_review_count").alias("avg_reviews_per_user")
    # )
    
    # -------------------------------
    # 6. Join results
    # -------------------------------
    # final_q3 = band_stats.join(avg_user_reviews, on="rating_band", how="inner")
    
    # -------------------------------
    # 7. Sort by total reviews descending
    # -------------------------------
    # final_q3 = final_q3.orderBy(col("total_reviews").desc())
    
    # -------------------------------
    # 8. Showing results
    print("===== Q3 Results =====")
    band_stats.show(truncate=False)
    
    # 9. Saving output in csv for plotting
    band_stats.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task2/q3_results/")

    
    # Question 4: Time-Based Review Analysis ------------------------------------------------------------------------------------

    #  these columns exist:
    # year
    # year_period (Early Year / Late Year)
    
    # 2. Group by year and half-year
    year_half_stats = ratings_df.groupBy("year", "year_period").agg(
        count("*").alias("total_reviews"),
        avg("review_score").alias("avg_review_score")
    ).orderBy("year","year_period")
    
    # 3. Yearly total reviews 
    year_stats = ratings_df.groupBy("year").agg(
        count("*").alias("year_total_reviews")
    )
    
    # 4. Window for year-on-year analysis
    year_window = Window.orderBy("year")
    
    year_stats = year_stats.withColumn(
        "prev_year_reviews",
        lag("year_total_reviews").over(year_window)
    )
    
    # Year-on-year change
    year_stats = year_stats.withColumn(
        "yoy_change",
        col("year_total_reviews") - col("prev_year_reviews")
    )
    
    # Year-on-year percentage growth
    year_stats = year_stats.withColumn(
        "yoy_growth_pct",
        when(
            col("prev_year_reviews").isNotNull(),
            (col("yoy_change") / col("prev_year_reviews")) * 100
        )
    )
    
    # 5. Identify key years
    max_activity_year = year_stats.orderBy(col("year_total_reviews").desc()).first()
    
    max_growth_year = year_stats.orderBy(col("yoy_growth_pct").desc()).first()
    
    print("======================================")
    print("Year with highest review activity:", max_activity_year['year'],";  total number of reviews :",max_activity_year['year_total_reviews'])
    print("Year with highest growth:", max_growth_year['year'], ";  % yoy growth :", max_growth_year['yoy_growth_pct'])
    print("======================================")
    
    # 6. Showing sample results 
    print("===== Year-Half Aggregation =====")
    year_half_stats.orderBy("year", "year_period").show(10, truncate=False)
    
    print("===== Yearly Growth Stats =====")
    year_stats.orderBy("year").show(10, truncate=False)
    
    # 7. Exporting csv for plotting 
    
    # (A) Early vs Late year analysis
    year_half_stats.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task2/q4_halfyear_results/")
    
    # (B) Yearly trend + growth
    year_stats.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task2/q4_yearly_results/")

    
    # Question 5: Category Popularity and Review Behaviour  ------------------------------------------------------------------------------------------
    
    
    # 1. Join ratings with books dataset
    books_ratings_df = ratings_df.join(
        books_df,
        on="id",
        how="inner"
    )
    
    
    # 2. Compute category statistics
    category_stats = books_ratings_df.groupBy("categories").agg(
        count("*").alias("total_reviews"),
        avg("review_score").alias("avg_review_score"),
        countDistinct("user_id").alias("distinct_users"),
        countDistinct("id").alias("distinct_books")
    )
    
    # 3. Average reviews per book
    category_stats = category_stats.withColumn(
        "avg_reviews_per_book",
        col("total_reviews") / col("distinct_books")
    )
    
    # 4. Sorting and get TOP 10 
    top10_reviews = category_stats.orderBy(col("total_reviews").desc()).limit(10)
    top10_avg_score = category_stats.orderBy(col("avg_review_score").desc(),col("total_reviews").desc()).limit(10)
    top10_avg_reviews_perBook = category_stats.orderBy(col("avg_reviews_per_book").desc(),col("total_reviews").desc()).limit(10)
    
    
    # 5. Show results 
    print("===== Top 10 Categories by Review Activity =====")
    top10_reviews.show(truncate=False)
    
    # 6. EXPORTING CSVs 
    
    # A) Top 10 categories by total reviews
    top10_reviews.coalesce(1) \
        .write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task2/q5_total_reviews/")
    
    # B) Top 10 categories by average review score
    top10_avg_score.coalesce(1) \
        .write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task2/q5_avg_review_score/")
    
    # C) Top 10 categories by avg reviews per book
    top10_avg_reviews_perBook.coalesce(1) \
        .write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task2/q5_avg_reviews_per_book/")

    
    # Question 6: Top Books by Rating and Popularity -------------------------------------------------------------------------------------------

    # 1. Join ratings and books data
    books_ratings_df = ratings_df.join(
        books_df,
        on="id",
        how="inner"
    )

    # 2. Group by title and compute metrics
    book_stats = books_ratings_df.groupBy("title").agg(
        count("*").alias("total_reviews"),
        avg("review_score").alias("avg_review_score"),
        countDistinct("user_id").alias("distinct_users")
    )

    # 3. Filter books with at least 50 reviews
    book_stats = book_stats.filter(col("total_reviews") >= 50)

    # 4. Sort by average review score 
    #    and then by total reviews 
    book_stats = book_stats.orderBy(
        col("avg_review_score").desc(),
        col("total_reviews").desc()
    )

    # 5. Get Top 10 books
    top10_books = book_stats.limit(10)

    print("===== Top 10 Books by Rating and Popularity =====")
    top10_books.show(10,truncate=False)

    # 6. Export results for report
    top10_books.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task2/q6_results/")


    # Question 7a: Reviewer Activity --------------------------------------------------------------------------------------------------

    # 1. Count reviews per user
    user_review_counts = ratings_df.groupBy("user_id").agg(
        count("*").alias("review_count")
    )

    # 2. Categorise users
    user_review_counts = user_review_counts.withColumn(
        "reviewer_type",
        when(col("review_count") > 50, "Frequent Reviewer")
        .otherwise("Infrequent Reviewer")
    )

    # 3. Top 10 most active reviewers
    top10_reviewers = user_review_counts.orderBy(
        col("review_count").desc()
    ).limit(10)

    print("===== Top 10 Most Active Reviewers =====")
    top10_reviewers.show(truncate=False)

    # 4. Count Frequent vs Infrequent reviewers
    reviewer_distribution = user_review_counts.groupBy("reviewer_type").agg(
        count("*").alias("num_reviewers")
    )

    print("===== Reviewer Distribution =====")
    reviewer_distribution.show(truncate=False)


    # 5. Export results
    top10_reviewers.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task2/q7a_top_reviewers/")

    reviewer_distribution.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task2/q7a_distribution/")



    # Question 7b: Reviewer Activity per Category

    # 1. Join ratings with books
    books_ratings_df = ratings_df.join(
        books_df,
        on="id",
        how="inner"
    )

    # 3. Count distinct reviewers per category
    category_reviewer_counts = books_ratings_df.groupBy("categories").agg(
        countDistinct("user_id").alias("distinct_reviewers")
    )

    # 4. Top 5 categories by reviewers
    top5_categories = category_reviewer_counts.orderBy(
        col("distinct_reviewers").desc()
    ).limit(5)

    print("===== Top 5 Categories by Number of Reviewers =====")
    top5_categories.show(10,truncate=False)

    # 5. Export results for chart
    category_reviewer_counts.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task2/q7b_category_reviewers/")

    top5_categories.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task2/q7b_top5_categories/")

    

    
    
    spark.stop()