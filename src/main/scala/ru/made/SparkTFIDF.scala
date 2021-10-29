package ru.made
import org.apache.spark.sql.functions
import org.apache.spark.sql.expressions._
import org.apache.spark.sql._
import org.apache.spark.sql.functions._

object SparkTFIDF {
  def main(args: Array[String]): Unit = {
    // Создает сессию спарка
    val spark = SparkSession.builder()
      // адрес мастера
      .master("local[*]")
      // имя приложения в интерфейсе спарка
      .appName("made-demo")
      // взять текущий или создать новый
      .getOrCreate()

    // синтаксический сахар для удобной работы со спарк
    import spark.implicits._
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("C:/Homeworks/Lesson-6-spark-sbt-example/data/tripadvisor_hotel_reviews.csv")

    val dfLower = df
      .withColumn("ReviewLower", lower(col("Review")))
    val dfLowerCleared = dfLower
      .withColumn("ReviewLowerClear",
        split(functions.regexp_replace(col("ReviewLower"), "[^a-z0-9 ]", ""), " "))
    val dfPrepared = dfLowerCleared.withColumn("index", monotonically_increasing_id())
    val dfPrepared2 = dfPrepared
      .withColumn("WordСount", size(col("ReviewLowerClear")))
    val dfWords= dfPrepared2
      .withColumn("Word", explode(col("ReviewLowerClear")))

    val dfReviewWordCnt = dfWords
      .groupBy("index", "Word")
      .agg(
        count("ReviewLowerClear") as "TermCount",
        first("WordСount") as "WordСount")
    val dfWordTF = dfReviewWordCnt.withColumn("TermFreq", col("TermCount") / col("WordСount"))

    val window = Window.orderBy(col("DocFreq").desc)
    val dfWordDF = dfWords
      .groupBy("Word")
      .agg(countDistinct("index") as "DocFreq")
      .withColumn("row", row_number.over(window))
      .where(col("row") < 100)
    val lenDf: Double = dfLowerCleared.count()
    def getIdf: UserDefinedFunction = {
      udf((DocFreq: Int) => {math.log(lenDf / (DocFreq.toDouble + 0.0001)) })
    }
    val dfWordIDF = dfWordDF
      .withColumn("IDF", getIdf(col("DocFreq")))
    val dfWordTfIdf = dfWordIDF
      .join(dfWordTF, Seq("Word"), "left")
      .withColumn("TFIDF", col("TermFreq") * col("IDF"))
    val denseTfIdf = dfWordTfIdf.groupBy("index")
      .pivot(col("Word"))
      .agg(first(col("TFIDF"), ignoreNulls = true))
    denseTfIdf.show
    denseTfIdf
      .coalesce(1)
      .write
      .option("header", "true")
      .option("sep", ",")
      .csv("C:/Homeworks/Lesson-6-spark-sbt-example/data/hw4_result.csv")
  }
}
