import sys, os, time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, lit, desc
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from graphframes import GraphFrame
from pyspark import *
from pyspark.sql.functions import col, lit, when
from pyspark.sql import *
from pyspark.sql.types import *
import graphframes
from graphframes import *
from pyspark.sql.functions import col, least, lit

if __name__ == "__main__":

    spark = SparkSession \
        .builder \
        .appName("Task3") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    userId = "ec25240"
    
    # sqlContext = SQLContext(spark)
    spark.sparkContext.setCheckpointDir(f"s3a://bdp-student-ec25240/task3/checkpoints")

    taxi_path = "s3a://module-big-data-processing/nyc_taxi/yellow_tripdata/2023/*.csv"
    zone_path = "s3a://module-big-data-processing/nyc_taxi/taxi_zone_lookup.csv"

    # QUESTION 1. SCHEMAS ---------------------------------------------------------------------------------------------------------------------------

    taxiSchema = StructType([
        StructField("tpep_pickup_datetime", StringType(), True),
        StructField("tpep_dropoff_datetime", StringType(), True),
        StructField("passenger_count", StringType(), True),
        StructField("trip_distance", StringType(), True),
        StructField("PULocationID", StringType(), True),
        StructField("DOLocationID", StringType(), True),
        StructField("payment_type", StringType(), True),
        StructField("fare_amount", StringType(), True),
        StructField("extra", StringType(), True),
        StructField("mta_tax", StringType(), True),
        StructField("tip_amount", StringType(), True),
        StructField("tolls_amount", StringType(), True),
        StructField("total_amount", StringType(), True),
        StructField("congestion_surcharge", StringType(), True),
        StructField("airport_fee", StringType(), True),
        StructField("taxi_type", StringType(), True)
    ])

    zoneSchema = StructType([
        StructField("LocationID", StringType(), True),
        StructField("Borough", StringType(), True),
        StructField("Zone", StringType(), True),
        StructField("service_zone", StringType(), True)
    ])

    # 2. LOAD DATA

    taxiDF = spark.read.option("header", True).schema(taxiSchema).csv(taxi_path)
    zoneDF = spark.read.option("header", True).schema(zoneSchema).csv(zone_path)

    # 3. DATA CLEANING 

    # cast required fields to integer
    taxiClean = taxiDF \
        .withColumn("PULocationID", col("PULocationID").cast(IntegerType())) \
        .withColumn("DOLocationID", col("DOLocationID").cast(IntegerType())) \
        .withColumn("trip_distance", col("trip_distance").cast(DoubleType())) \
        .withColumn("total_amount", col("total_amount").cast(DoubleType())) \
        .dropna(subset=["PULocationID", "DOLocationID"])

    zoneClean = zoneDF \
        .withColumn("LocationID", col("LocationID").cast(IntegerType()))

    
    # 4. BUILD VERTICES

    vertices = zoneClean.select(
        col("LocationID").alias("id"),
        "Borough",
        "Zone",
        "service_zone"
    )

    
    # 5. BUILD EDGES

    edges = taxiClean.select(
        col("PULocationID").alias("src"),
        col("DOLocationID").alias("dst"),
        col("total_amount").alias("trip_total"),
        col("trip_distance")
    )

    # 6. CREATE GRAPH

    g = GraphFrame(vertices, edges)

    # 7. PREVIEWING TABLES

    print("===== VERTICES SAMPLE =====")
    vertices.show(10, truncate=False)
    time.sleep(10)
    vertices.limit(10).coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q1_vertices/")

    print("===== EDGES SAMPLE =====")
    edges.show(10, truncate=False)
    time.sleep(10)
    edges.limit(10).coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q1_edges/")
    

    # 8. BASIC STATISTICS 

    raw_trip_count = taxiClean.count()
    vertex_count = vertices.count()
    edge_count = edges.count()

    print("===== SUMMARY =====")
    print("Raw Trip Count:", raw_trip_count)
    print("Number of Vertices:", vertex_count)
    print("Number of Edges:", edge_count)
    time.sleep(10)

    print("Null PULocationID:", taxiDF.filter(col("PULocationID").isNull()).count())
    print("Null DOLocationID:", taxiDF.filter(col("DOLocationID").isNull()).count())

    # Question 2 ------------------------------------------------------------------------------------------------------------------------
    
    # Q2a. Connected Components
    
    print("=== Running Connected Components ===")
    
    # Compute connected components
    components = g.connectedComponents(algorithm="graphx")
    
    # Each row contains vertex id and component id
    component_stats = components.groupBy("component") \
        .count() \
        .withColumnRenamed("count", "component_size")
    
    # Total number of components
    num_components = component_stats.count()
    
    print(f"Total Connected Components: {num_components}")
    
    # Top 5 component sizes
    top5_components = component_stats.orderBy(desc("component_size")).limit(5)
    
    print("=== Top 5 Component Sizes ===")
    top5_components.show(5, truncate=False)
    time.sleep(10)
    
    # Export component statistics
    component_stats.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q2_components/")
    
    # Export top5 components for screenshot
    top5_components.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q2_top5_components/")
    
    
    # Q2b. Degree Analysis
    
    print("=== Computing Degree Metrics ===")
    
    # In-degree
    in_deg = g.inDegrees.withColumnRenamed("inDegree", "in_degree")
    
    # Out-degree
    out_deg = g.outDegrees.withColumnRenamed("outDegree", "out_degree")
    
    # Total degree = in + out
    degree_df = in_deg.join(out_deg, "id", "outer").na.fill(0)
    
    degree_df = degree_df.withColumn(
        "total_degree",
        col("in_degree") + col("out_degree")
    )

    degree_with_info = degree_df.join(vertices, "id")
    
    
    # Top 10 zones by total degree
    top10_degree = degree_with_info.orderBy(desc("total_degree")).limit(10)
    
    print("=== Top 10 Zones by Total Degree ===")
    top10_degree.show(10, truncate=False)
    time.sleep(10)
    
    # Export degree statistics
    degree_df.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q2_degree_stats/")
    
    # Export top 10 degree zones
    top10_degree.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q2_top10_degree/")
    
    
    # Triangle Count Analysis
    
    print("=== Computing Triangle Count ===")
    
    triangle_counts = g.triangleCount(storage_level=StorageLevel.MEMORY_AND_DISK)
    
    # Select relevant columns
    triangle_df = triangle_counts.select(
        col("id"),
        col("Borough"),
        col("Zone"),
        col("service_zone"),
        col("count").alias("triangle_count")
    )
    
    # Top 10 zones by triangle count
    top10_triangles = triangle_df.orderBy(desc("triangle_count")).limit(10)
    
    print("=== Top 10 Zones by Triangle Count ===")
    top10_triangles.show(10, truncate=False)
    time.sleep(10)
    
    # Export triangle counts
    triangle_df.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q2_triangle_counts/")
    
    # Export top 10 triangle zones
    top10_triangles.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q2_top10_triangles/")

    # Question 3: Shortest Path & Reachability -------------------------------------------------------------------------------------------
    
    print("=== Q3: Shortest Path & Reachability ===")
    
    filtered_edges = edges.filter(
        (col("trip_distance") > 0) & (col("trip_total") > 0)
    )
    
    # Recreate filtered graph
    g_filtered = GraphFrame(vertices, filtered_edges)
    
    # Cache for performance
    g_filtered.vertices.cache()
    g_filtered.edges.cache()
    
    print("Filtered edges count:", g_filtered.edges.count())
    time.sleep(10)
    
    # Step 2: Choose start and goal zones
    
    # start_id = 230     # Example: Manhattan (Alphabet City)
    # goal_id = 132    # Example: JFK Airport (common ID)
    
    # print(f"Start Zone ID: {start_id}")
    # print(f"Goal Zone ID: {goal_id}")

    print("=== Sample Zones ===")
    vertices.select("id", "Borough", "Zone",).show(10, truncate=False)
    time.sleep(10)
    
    # You should verify IDs manually from output
    start_id = 230   # Example Manhattan zone 
    goal_id = 132    # Example JFK Airport 
    
    print(f"Start Zone ID: {start_id}")
    print(f"Goal Zone ID: {goal_id}")
    
    
    
    # Step 3: BFS bounded depth <= 4
    
    print("=== Running BFS (max depth = 4) ===")
    
    bfs_result = g_filtered.bfs(
        fromExpr=f"id = {start_id}",
        toExpr=f"id = {goal_id}",
        maxPathLength=4
    )
    
    # Show at least one valid path
    print("=== BFS Path Result ===")
    bfs_result.show(5, truncate=False)
    time.sleep(10)
    
    # # Save BFS output for report
    # bfs_result.coalesce(1).write \
    #     .mode("overwrite") \
    #     .option("header", True) \
    #     .csv("s3a://bdp-student-ec25240/sample_data/task3/q3_bfs_paths/")
    
    
    # Step 4: Shortest Paths (multi-destination reachability)
    
    print("=== Running shortestPaths (landmark = goal zone) ===")
    
    shortest_paths = g_filtered.shortestPaths(landmarks=[goal_id])
    
    # Extract distance to goal
    # distances is a map: {goal_id: distance}
    sp_df = shortest_paths.select(
        col("id"),
        col("distances").getItem(goal_id).alias("distance_to_goal")
    )
    
    # Join with zone info for readability
    sp_df = sp_df.join(vertices, "id")
    
    # Show 10 sample results
    print("=== Sample Shortest Path Distances ===")
    # sp_sample = sp_df.orderBy("distance_to_goal").limit(10)
    sp_sample = sp_df.orderBy(col("id").asc()).limit(10)
    sp_sample.show(10, truncate=False)
    time.sleep(10)
    
    # Save results
    sp_df.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q3_shortest_paths/")
    
    sp_sample.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q3_shortest_paths_sample/")


    # Question 4: PageRank Analysis ----------------------------------------------------------------------------------------------------------
    
    print("=== Q4: PageRank Analysis ===")
    
    
    # Q4a: Classic PageRank
    
    print("=== Q4a: Classic PageRank ===")
    
    # Deduplicate edges
    edges_dedup = edges.select("src", "dst").dropDuplicates()
    
    print("Deduplicated edges count:", edges_dedup.count())
    
    g_dedup = GraphFrame(vertices, edges_dedup)
    
    pr_results = g_dedup.pageRank(
        resetProbability=0.15,
        tol=0.01
    )
    
    pr_vertices = pr_results.vertices
    
    top10_pagerank = pr_vertices.orderBy(desc("pagerank"))
    
    print("=== Top 10 Zones (Classic PageRank) ===")
    top10_pagerank.select("id", "Borough", "Zone","service_zone", "pagerank").show(10, truncate=False)
    time.sleep(10)
    
    # Saving csv
    pr_vertices.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q4a_pagerank/")
    
    top10_pagerank.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q4a_top10/")
    
    
    # Q4b: Frequency-Aware PageRank 
    
    print("=== Q4b: Frequency-Aware PageRank ===")
    
    # Step 1: Compute trip frequency per route
    freq_edges = edges.groupBy("src", "dst") \
        .count() \
        .withColumnRenamed("count", "trip_count")
    
    print("Frequency edges count:", freq_edges.count())
    
    # Step 2: Expand edges based on frequency (simulate weights)
    # Each (src, dst) appears multiple times proportional to trip_count
    
    # # Cap frequency to avoid explosion
    # freq_edges_limited = freq_edges.withColumn(
    #     "trip_count",
    #     least(col("trip_count"), lit(50))   # cap at 50
    # )
    
    # expanded_edges = freq_edges_limited.selectExpr(
    #     "src",
    #     "dst",
    #     "explode(array_repeat(1, CAST(trip_count AS INT))) as dummy"
    # ).select("src", "dst")
    
    # print("Expanded edges count (after applying frequency):", expanded_edges.count())
    
    # Step 3: Create graph using expanded edges
    g_freq = GraphFrame(vertices, freq_edges)
    
    # Step 4: Run PageRank
    pr_freq_results = g_freq.pageRank(
        resetProbability=0.15,
        tol=0.01
    )
    
    pr_freq_vertices = pr_freq_results.vertices
    
    # Step 5: Top 10 zones in descending order
    top10_freq_pagerank = pr_freq_vertices.orderBy(desc("pagerank")).limit(10)
    
    print("=== Top 10 Zones (Frequency-Aware PageRank) ===")
    top10_freq_pagerank.select("id", "Borough", "Zone","service_zone", "pagerank").show(10, truncate=False)
    time.sleep(10)
    print("=== Top 10 Zones (Frequency-Aware PageRank) ===")
    
    # Save results
    pr_freq_vertices.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q4b_pagerank/")
    
    top10_freq_pagerank.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q4b_top10/")


    # Question 5: Zone-Level Community Detection --------------------------------------------------------------------------------------------------
    
    print("=== Q5: Label Propagation Community Detection ===")
    
    # Step 1: Use deduplicated graph
    
    edges_lpa = edges.select("src", "dst").dropDuplicates()
    g_lpa = GraphFrame(vertices, edges_lpa)
    
    # Step 2: Run Label Propagation
    
    print("=== Running Label Propagation ===")
    
    lpa_result = g.labelPropagation(maxIter=5)
    
    # keep ONLY required columns to avoid hidden duplicates
    lpa_clean = lpa_result.select("id", "label")
    
    # Step 3: Join with vertex metadata 
    
    # Rename columns BEFORE join to avoid ambiguity permanently
    vertices_clean = vertices.select(
        col("id"),
        col("Borough").alias("borough_name"),
        col("Zone").alias("zone_name"),
        col("service_zone")
    )
    
    communities = lpa_clean.join(vertices_clean, "id")
    
    print("=== Sample Community Assignment ===")
    communities.show(10, truncate=False)
    time.sleep(10)
    communities.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q5_sample_communities/")
    
    # Step 4: Number of communities
    
    num_communities = communities.select("label").distinct().count()
    print("Number of detected communities:", num_communities)
    
    # Step 5: Top 5 largest communities
    
    community_sizes = communities.groupBy("label") \
        .count() \
        .withColumnRenamed("count", "community_size")
    
    top5_communities = community_sizes.orderBy(desc("community_size")).limit(5)
    
    print("=== Top 5 Largest Communities ===")
    top5_communities.show(5, truncate=False)
    time.sleep(10)
    top5_communities.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q5_top_communtities/")
    
    # Step 6: Borough composition per community 
    
    borough_composition = communities.groupBy("label", "borough_name") \
        .count() \
        .orderBy("label", desc("count"))
    
    print("=== Community → Borough Composition ===")
    borough_composition.show(20, truncate=False)
    time.sleep(10)
    
    # Saving csv
    borough_composition.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q5_borough_composition/")
    
    # Step 7: Top hubs in largest community
    
    largest_community_id = top5_communities.first()["label"]
    
    largest_nodes = communities.filter(col("label") == largest_community_id)
    
    # bring degree column 
    degree_only = degree_with_info.select("id", "total_degree")
    
    hubs = largest_nodes.join(degree_only, "id")
    
    top5_hubs = hubs.orderBy(desc("total_degree")).limit(5)
    
    print("=== Top 5 Hubs in Largest Community ===")
    top5_hubs.select(
        "id", "borough_name", "zone_name","service_zone", "total_degree"
    ).show(5, truncate=False)
    time.sleep(10)
    
    # Save hubs
    top5_hubs.coalesce(1).write \
        .mode("overwrite") \
        .option("header", True) \
        .csv("s3a://bdp-student-ec25240/task3/q5_top5_hubs/")
            
        
    spark.stop()
    sys.exit(0)