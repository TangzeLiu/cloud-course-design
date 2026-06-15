from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, round as spark_round

DATA_PATH = "file:///opt/spark/work/douban_movies.csv"

spark = (
    SparkSession.builder
    .appName("DoubanMoviesCleaning")
    .getOrCreate()
)

print("===== A-1 DATA CLEANING: DOUBAN MOVIES =====")
print("DATA_PATH =", DATA_PATH)

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    .csv(DATA_PATH)
)

print("===== 1. Schema =====")
df.printSchema()

print("===== 2. First 5 Rows =====")
df.show(5, truncate=False)

raw_count = df.count()
print("===== 3. Raw Row Count =====")
print(f"Raw row count: {raw_count}")

print("===== 4. Missing Value Ratio =====")
missing_exprs = []
for field in df.schema.fields:
    c = field.name
    dtype = field.dataType.simpleString()

    if dtype in ("double", "float"):
        missing_cond = col(c).isNull() | isnan(col(c))
    else:
        missing_cond = col(c).isNull() | (col(c).cast("string") == "")

    missing_exprs.append(
        spark_round(
            count(when(missing_cond, c)) / raw_count * 100,
            2
        ).alias(c + "_missing_percent")
    )

df.select(missing_exprs).show(truncate=False)

print("===== 5. Cleaning Strategy =====")
print("Strategy A: Drop rows with missing critical fields: title, year, rating_score.")
print("Strategy B: Fill missing descriptive fields genres/countries/directors/summary with default values.")

clean_df = df.dropna(subset=["title", "year", "rating_score"])

clean_df = clean_df.fillna({
    "original_title": "Unknown",
    "genres": "Unknown",
    "countries": "Unknown",
    "directors": "Unknown",
    "summary": ""
})

clean_df = (
    clean_df
    .withColumn("movie_id", col("movie_id").cast("long"))
    .withColumn("year", col("year").cast("int"))
    .withColumn("rating_score", col("rating_score").cast("double"))
    .withColumn("rating_count", col("rating_count").cast("long"))
    .withColumn("collect_count", col("collect_count").cast("long"))
)

clean_count = clean_df.count()

print("===== 6. Row Count Before and After Cleaning =====")
print(f"Before cleaning row count: {raw_count}")
print(f"After cleaning row count: {clean_count}")
print(f"Removed row count: {raw_count - clean_count}")

print("===== 7. Missing Value Ratio After Cleaning =====")
missing_exprs_after = []
for field in clean_df.schema.fields:
    c = field.name
    dtype = field.dataType.simpleString()

    if dtype in ("double", "float"):
        missing_cond = col(c).isNull() | isnan(col(c))
    else:
        missing_cond = col(c).isNull() | (col(c).cast("string") == "")

    missing_exprs_after.append(
        spark_round(
            count(when(missing_cond, c)) / clean_count * 100,
            2
        ).alias(c + "_missing_percent")
    )

clean_df.select(missing_exprs_after).show(truncate=False)

print("===== 8. Basic Statistics: mean/std/min/max =====")
clean_df.select(
    "year",
    "rating_score",
    "rating_count",
    "collect_count"
).describe().show(truncate=False)

print("===== 9. Cleaned Sample Rows =====")
clean_df.select(
    "movie_id",
    "title",
    "year",
    "rating_score",
    "rating_count",
    "genres",
    "countries",
    "directors"
).show(10, truncate=False)

spark.stop()