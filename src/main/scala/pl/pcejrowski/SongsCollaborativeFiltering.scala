package pl.pcejrowski

import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object SongsCollaborativeFiltering {

  def main(args: Array[String]): Unit = {
    val conf: SparkConf = new SparkConf()
      .setAppName("songs-collaborative-filtering")
      .setMaster("local[4]")
    val sc: SparkContext = new SparkContext(conf)

    val sourceCSV: RDD[Array[String]] = sc.textFile("songsDataset.csv")
      .map(_.split(","))

    val ratings: RDD[Rating] = sourceCSV
      .map(r => Rating(r(0).toInt, r(1).toInt, r(2).toDouble))

    ALS
      .train(ratings, 8, 6)
      .recommendProductsForUsers(3)
      .flatMap { case (userId, recommendation) => recommendation.map(r => (r.product, userId)) }
      .map { case (userId, songId) => s"Song $songId is recommended for user $userId" }
      .saveAsTextFile("recommendations.txt")

    Console.in.read.toChar // take a look at localhost:4040 Spark console
  }
}