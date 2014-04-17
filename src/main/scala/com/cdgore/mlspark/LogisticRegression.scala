/**
 * Logistic regression classifier
 * @author cdgore
 * 2013-10-17
 */
package com.cdgore.mlspark

import org.apache.spark
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import java.io.FileInputStream
import java.io.IOException
import java.util.Properties

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.NullWritable
import org.apache.hadoop.io.Text
import org.apache.hadoop.mapred.TextOutputFormat

import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions._

/**
 * @author cdgore
 *
 */
object LogisticRegression extends App {
//  def parseData(line: String): (String, DoubleMatrix) = {
//    return (line.split(",")(0), new DoubleMatrix(line.split(",")(1).split("\t").map(_.toDouble)))
//  }

  def parseTrainData(line: String): (String, DoubleMatrix) = {
    return (line.split("\t")(2), new DoubleMatrix(line.split("\t").slice(3, line.length).map(_.toDouble)))
  }

  def parsePredictData(line: String): (String, String, DoubleMatrix) = {
    return (line.split("\t")(2), line.split("\t")(0), new DoubleMatrix(line.split("\t").slice(3, line.length).map(_.toDouble)))
  }

  def saveAsCSV(rdd: spark.rdd.RDD[(String, String)], path: String, delim: String = ",") {
    rdd.map(x => (NullWritable.get(), new Text(x._1 + delim + x._2)))
      .saveAsHadoopFile[TextOutputFormat[NullWritable, Text]](path)
  }

  def SaveRDDAsHadoopFile(rdd: spark.rdd.RDD[(String, String)], path: String) {
    rdd.map(x => (new Text(x._1), new Text(x._2)))
      .saveAsHadoopFile[TextOutputFormat[Text, Text]](path)
  }

//  def SaveRDDAsHadoopFile(rdd: spark.rdd.RDD[String], path: String) {
//    rdd.map(x => (NullWritable.get(), new Text(x.toString())))
//      .saveAsHadoopFile[TextOutputFormat[NullWritable, Text]](path)
//  }
  
  // Train
  def train(trainData: spark.rdd.RDD[(Int, DoubleMatrix)], sc: SparkContext, learningRateAlpha: Double, l2Lambda: Double, maxIterations: Int): DoubleMatrix = {
    // Initialize weight vector
    val numFeatures = trainData.first._2.length
    var W = DoubleMatrix.randn(numFeatures).muli(2).subi(1)

    for (i <- 1 to maxIterations) {
      println("Iteration number: " + i + "/" + maxIterations)
      // Calculate gradient
      val gradient = trainData.map { p =>
        preprocessFeatures(p._2).mul(p._1 - (1 / (1 + exp(W.dot(preprocessFeatures(p._2))))))
      }.reduce(_ addi _)
      // Apply gradient to weight vector in update step
      W = W.add((W.mul(-1 * l2Lambda).add(gradient)).mul(learningRateAlpha))
    }
    // Return learned weights
    return W
  }

  // Predict
  def predict(predictData: spark.rdd.RDD[(String, String, org.jblas.DoubleMatrix)], W: DoubleMatrix): spark.rdd.RDD[(String, String)] = {
    predictData.map {
      case(l, uid, x) => (uid, math.round(1 / (1 + exp(W.dot(preprocessFeatures(x))))).toString)
    }
  }
  
  // Shrink the input features by taking the log, smoothening, and dividing by 10
  def preprocessFeatures(x: DoubleMatrix): DoubleMatrix = {
    return new DoubleMatrix(x.toArray.map(x => math.log(math.abs(x)+1.001)/10))
  }
  
  override def main(args: Array[String]) {
    var master = "local[2]"
    if (System.getProperty("spark.MASTER") != null)
      master = System.getProperty("spark.MASTER")
    else if (System.getProperty("sparkEnvPropertiesFile") != null) {
      val sparkEnvProps = new Properties()
      try {
        sparkEnvProps.load(new FileInputStream(System.getProperty("sparkEnvPropertiesFile")))
        if (sparkEnvProps.getProperty("spark.MASTER") != null)
          master = sparkEnvProps.getProperty("spark.MASTER")
        else
          System.err.println("ERROR: Unable to read property 'spark.MASTER'")
      } catch {
        case e: Exception => println(e)
      }
    }
    val jars = System.getProperty("jarPath") match {
      case x:String => x.split(',').toList
      case _ => List()
    }
    val sc = new SparkContext(master,"Gaussian Naive Bayes classifier", System.getProperty("spark.home"),
        jars)
    val conf = new Configuration()
    if (System.getenv("HADOOP_HOME") != null) {
      try {
        conf.addResource(new Path(System.getenv("HADOOP_HOME") + "/conf/core-site.xml"))
      } catch {
        case e: Exception => println(e)
      }
    } else if (System.getProperty("hadoopConfFile") != null)
      conf.addResource(new Path(System.getProperty("hadoopConfFile")))
    else
      System.err.println("WARNING: Cannot find core-site.xml and property 'hadoopConfFile' not specified")
    try {
      sc.hadoopConfiguration.set("fs.s3n.awsAccessKeyId", conf.get("fs.s3n.awsAccessKeyId"))
      sc.hadoopConfiguration.set("fs.s3n.awsSecretAccessKey", conf.get("fs.s3n.awsSecretAccessKey"))
    } catch {
      case e: Exception => println(e)
    }
    if (System.getProperty("fs.s3n.awsAccessKeyId") != null)
      sc.hadoopConfiguration.set("fs.s3n.awsAccessKeyId", System.getProperty("fs.s3n.awsAccessKeyId"))
    if (System.getProperty("fs.s3n.awsSecretAccessKey") != null)
      sc.hadoopConfiguration.set("fs.s3n.awsSecretAccessKey", System.getProperty("fs.s3n.awsSecretAccessKey"))
    if (System.getProperty("trainFile") == null || System.getProperty("predictFile") == null || System.getProperty("predictionsOutputFile") == null)
      throw new IOException("ERROR: Must specify properties 'trainFile', 'predictFile', and 'predictionsOutputFile'")
    System.getProperties().list(System.out)
    
    val learningRateAlpha = System.getProperty("learningRateAlpha").toDouble
    val l2Lambda = System.getProperty("l2Lambda").toDouble
    val maxIterations = System.getProperty("maxIterations").toInt
    
    val trainFile = sc.textFile(System.getProperty("trainFile")).persist(spark.storage.StorageLevel.DISK_ONLY)
    val predictFile = sc.textFile(System.getProperty("predictFile")).persist(spark.storage.StorageLevel.DISK_ONLY)
    
//    val trainDataTmp = trainFile.map(parseTrainData _)
    // Normalize target values to 0 or 1
    val trainData = trainFile.map(parseTrainData _).map {
      case (k, v) => (k.toInt match {
        case n if n > 0 => 1
        case n if n == 0 => 0
        case _ => 0
      }, v)
    }.persist(spark.storage.StorageLevel.DISK_ONLY) //.persist(spark.storage.StorageLevel.MEMORY_AND_DISK)
    val predictData = predictFile.map(parsePredictData _).persist(spark.storage.StorageLevel.DISK_ONLY)//.persist(spark.storage.StorageLevel.MEMORY_AND_DISK)
    
    val learnedWeights = train(trainData, sc, learningRateAlpha, l2Lambda, maxIterations)
    val predictedClasses = predict(predictData, learnedWeights)
    SaveRDDAsHadoopFile(predictedClasses, System.getProperty("predictionsOutputFile"))
//    predictedClasses.saveAsTextFile(System.getProperty("predictionsOutputFile"))
//    predictedClasses.saveAsTextFile(System.getProperty("predictionsOutputFile"))
  }
}
