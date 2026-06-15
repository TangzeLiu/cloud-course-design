import os
import time
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, regexp_replace, avg, count, round as spark_round

DATA_PATH_LOCAL = "/opt/spark/work/douban_movies.csv"
DATA_PATH_SPARK = "file:///opt/spark/work/douban_movies.csv"

import sys
mode = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PERF_MODE", "spark")

def run_pandas():
    print("===== A-3 Pandas Performance Test =====")
    print("Query: GROUP BY genres, count movies and average rating_score")

    start = time.time()

    df = pd.read_csv(DATA_PATH_LOCAL)

    df = df.dropna(subset=["title", "year", "rating_score"])
    df["genres"] = df["genres"].fillna("Unknown")
    df["rating_score"] = pd.to_numeric(df["rating_score"], errors="coerce")
    df = df.dropna(subset=["rating_score"])

    df["genre_item"] = df["genres"].astype(str).str.replace("，", "/", regex=False).str.split("/")
    exploded = df.explode("genre_item")
    exploded = exploded[exploded["genre_item"].notna() & (exploded["genre_item"] != "")]

    result = (
        exploded.groupby("genre_item")
        .agg(
            movie_count=("movie_id", "count"),
            avg_rating=("rating_score", "mean")
        )
        .reset_index()
        .sort_values("movie_count", ascending=False)
        .head(10)
    )

    result["avg_rating"] = result["avg_rating"].round(2)

    end = time.time()
    elapsed = end - start

    print(result.to_string(index=False))
    print(f"PERF_RESULT pandas_seconds={elapsed:.4f}")


def run_spark():
    print("===== A-3 PySpark Performance Test =====")
    print("Query: GROUP BY genres, count movies and average rating_score")

    spark = (
        SparkSession.builder
        .appName("DoubanPerformanceTest")
        .getOrCreate()
    )

    start = time.time()

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("multiLine", True)
        .option("quote", '"')
        .option("escape", '"')
        .csv(DATA_PATH_SPARK)
    )

    clean = (
        df.dropna(subset=["title", "year", "rating_score"])
        .fillna({"genres": "Unknown"})
        .withColumn("rating_score", col("rating_score").cast("double"))
    )

    genre_df = (
        clean
        .withColumn("genre_item", explode(split(regexp_replace(col("genres"), "，", "/"), "/")))
        .where((col("genre_item").isNotNull()) & (col("genre_item") != ""))
    )

    result = (
        genre_df
        .groupBy("genre_item")
        .agg(
            count("*").alias("movie_count"),
            spark_round(avg("rating_score"), 2).alias("avg_rating")
        )
        .orderBy(col("movie_count").desc())
        .limit(10)
    )

    result.show(20, truncate=False)

    # 触发实际计算
    result.collect()

    end = time.time()
    elapsed = end - start

    print(f"PERF_RESULT spark_seconds={elapsed:.4f}")

    spark.stop()


if __name__ == "__main__":
    print(f"PERF_MODE={mode}")

    if mode == "pandas":
        run_pandas()
    else:
        run_spark()