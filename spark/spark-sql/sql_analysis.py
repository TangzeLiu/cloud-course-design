from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, split, regexp_replace, desc, row_number
from pyspark.sql.window import Window

DATA_PATH = "file:///opt/spark/work/douban_movies.csv"

spark = SparkSession.builder.appName("DoubanSparkSQLAnalysis").getOrCreate()

df = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .option("multiLine", True)
    .option("quote", '"')
    .option("escape", '"')
    .csv(DATA_PATH)
)

clean = (
    df.dropna(subset=["title", "year", "rating_score"])
    .fillna({"genres": "Unknown", "countries": "Unknown", "directors": "Unknown"})
    .withColumn("year", col("year").cast("int"))
    .withColumn("rating_score", col("rating_score").cast("double"))
    .withColumn("rating_count", col("rating_count").cast("long"))
    .withColumn("collect_count", col("collect_count").cast("long"))
)

movies = clean.select(
    "movie_id", "title", "year", "rating_score", "rating_count",
    "genres", "countries", "directors", "collect_count"
)
movies.createOrReplaceTempView("movies")

genre_df = (
    movies.withColumn("genre_item", explode(split(regexp_replace(col("genres"), "，", "/"), "/")))
    .where((col("genre_item").isNotNull()) & (col("genre_item") != ""))
)
genre_df.createOrReplaceTempView("movie_genres")

country_df = (
    movies.withColumn("country_item", explode(split(regexp_replace(col("countries"), "，", "/"), "/")))
    .where((col("country_item").isNotNull()) & (col("country_item") != ""))
)
country_df.createOrReplaceTempView("movie_countries")

print("===== A-2 Query 1: GROUP BY 类型聚合 =====")
q1 = spark.sql("""
SELECT genre_item,
       COUNT(*) AS movie_count,
       ROUND(AVG(rating_score), 2) AS avg_rating,
       ROUND(AVG(rating_count), 0) AS avg_rating_count
FROM movie_genres
GROUP BY genre_item
ORDER BY movie_count DESC
LIMIT 10
""")
q1.show(20, truncate=False)

print("===== A-2 Query 2: ORDER BY Top-N 高分电影 =====")
q2 = spark.sql("""
SELECT title, year, genres, countries, rating_score, rating_count
FROM movies
WHERE rating_count >= 10000
ORDER BY rating_score DESC, rating_count DESC
LIMIT 10
""")
q2.show(20, truncate=False)

print("===== A-2 Query 3: 时间维度趋势分析：按年份统计 =====")
q3 = spark.sql("""
SELECT year,
       COUNT(*) AS movie_count,
       ROUND(AVG(rating_score), 2) AS avg_rating,
       ROUND(AVG(rating_count), 0) AS avg_rating_count
FROM movies
WHERE year IS NOT NULL
GROUP BY year
ORDER BY year
""")
q3.show(80, truncate=False)

print("===== A-2 Query 4: JOIN 类型与国家统计 =====")
q4 = spark.sql("""
SELECT g.genre_item,
       c.country_item,
       COUNT(*) AS movie_count,
       ROUND(AVG(g.rating_score), 2) AS avg_rating
FROM movie_genres g
JOIN movie_countries c
  ON g.movie_id = c.movie_id
GROUP BY g.genre_item, c.country_item
HAVING movie_count >= 3
ORDER BY movie_count DESC, avg_rating DESC
LIMIT 20
""")
q4.show(30, truncate=False)

print("===== A-2 Query 5: 窗口函数：每年评分 Top 3 =====")
w = Window.partitionBy("year").orderBy(desc("rating_score"), desc("rating_count"))
q5 = (
    movies.where(col("year").isNotNull())
    .withColumn("rank_in_year", row_number().over(w))
    .where(col("rank_in_year") <= 3)
    .select("year", "rank_in_year", "title", "rating_score", "rating_count", "genres")
    .orderBy(desc("year"), "rank_in_year")
)
q5.show(80, truncate=False)

spark.stop()