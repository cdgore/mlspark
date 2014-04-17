/**
 * Gaussian Naive Bayes classifier
 * @author cdgore
 * 2013-08-21
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
object GaussianNaiveBayes extends App {
//  def parseData(line: String): (String, DoubleMatrix) = {
//    return (line.split(",")(0), new DoubleMatrix(line.split(",")(1).split("\t").map(_.toDouble)))
//  }

  def parseTrainData(line: String): (String, DoubleMatrix) = {
    (line.split("\t")(2), new DoubleMatrix(line.split("\t").slice(3, line.length).map(_.toDouble)))
  }

  def parsePredictData(line: String): (String, String, DoubleMatrix) = {
    (line.split("\t")(2), line.split("\t")(0), new DoubleMatrix(line.split("\t").slice(3, line.length).map(_.toDouble)))
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
  
  def getClassMeans(x: String, cMeans: Array[(String, DoubleMatrix)]): DoubleMatrix = {
    cMeans.filter{
      case(c, mu) => c.equals(x)
    }.map{
      case(a, b) => b
    }.reduce((x, y) => x)
  }

  def getSquaredDifferences(x: DoubleMatrix, mu: DoubleMatrix): DoubleMatrix = {
    pow(x.subi(mu), 2)
  }
  
  def applyGaussian(mu: DoubleMatrix, sigma: DoubleMatrix, x: DoubleMatrix): DoubleMatrix = {
    sqrt(pow(sigma, 2).mul(2.0 * scala.math.Pi)).rdiv(1).mul(exp(pow(x.sub(mu), 2).div(pow(sigma, 2).mul(2.0)).mul(-1.0)))
  }
  
  def sumLogLikelihoods(mu: DoubleMatrix, sigma: DoubleMatrix, x: DoubleMatrix): Double = {
    log(applyGaussian(mu, sigma, x)).sum
  }
  
  def generateClassPosteriorDistributions(x: DoubleMatrix, classParameters: Array[(String, ((org.jblas.DoubleMatrix, org.jblas.DoubleMatrix), Double))]): Array[(String, Double)] = {
    classParameters.map {
      case (cL, ((m, sig), pri)) => (cL, exp(sumLogLikelihoods(m, sig, x) + scala.math.log(pri)))
    }
  }
  
  // Train
  def train(trainData: spark.rdd.RDD[(String, DoubleMatrix)], sc: SparkContext): spark.rdd.RDD[(String, ((DoubleMatrix, DoubleMatrix), Double))] = {
    // Calculate class means
    val numFeatures = trainData.first._2.length
    val classMeans = trainData.combineByKey[(DoubleMatrix, Long)] (
        (v: DoubleMatrix) => (v, 1.toLong),
        (c: (DoubleMatrix, Long), v: DoubleMatrix) => (c._1.addi(v), c._2 + 1.toLong),
        (c1: (DoubleMatrix, Long), c2: (DoubleMatrix, Long)) => (c1._1.addi(c2._1), c1._2 + c2._2)
    ).map {
      case (k, (v, c)) => (k, v.div(c))
    }.collect()
    
    val classM = sc.parallelize(classMeans)
    
    // Variance smoothening, mostly for cases where there is no variance on a feature
    // within a class
    val epsilon = 1e-9
    
    // Calculate class standard deviations
    val classSDs = trainData.map {
      case (k, x) => (k, (getSquaredDifferences(x, getClassMeans(k, classMeans)), 1.toLong))
    }.combineByKey[(DoubleMatrix, Long)](
        (v: (DoubleMatrix, Long)) => v,
        (c: (DoubleMatrix, Long), v: (DoubleMatrix, Long)) => (c._1.addi(v._1), c._2 + v._2),
        (c1: (DoubleMatrix, Long), c2: (DoubleMatrix, Long)) => (c1._1.addi(c2._1), c1._2 + c2._2)
    ).map {
      case (k, (d, c)) => (k, sqrti(d.divi(c)).addi(epsilon))
    }.collect()
    val classS = sc.parallelize(classSDs)
    
    // Count the number of training samples
    val trainSampleCount = trainData.map {
      case (k, v) => (1)
    }.reduce(_ + _)
    
    // Calculate class priors
    val classPriors = trainData.map {
      case (k, v) => (k, 1)
    }.reduceByKey(_ + _).map {
      case (k, c) => (k, c.toDouble / trainSampleCount.toDouble)
    }.collect()
    val classP = sc.parallelize(classPriors)
    
    // Join and return class means, standard deviations, and priors
    classM.join(classS).join(classP)
  }

  // Predict
  def predict(predictData: spark.rdd.RDD[(String, String, org.jblas.DoubleMatrix)], classParamsArray: Array[(String, ((DoubleMatrix, DoubleMatrix), Double))]): spark.rdd.RDD[(String, String)] = {
    val classPosteriorDistributions = predictData.map {
      case (targetVar, rowId, features) => (rowId, generateClassPosteriorDistributions(features, classParamsArray))
    }

    classPosteriorDistributions.map {
      case (rowId, posteriorProbs: Array[(String, Double)]) => (rowId, posteriorProbs.foldLeft(("", -1.0))((b, a) => if (a._2 > b._2) (a._1, a._2) else (b._1, b._2)))
    }.map {
      case (rowId, (predictedClass, postProb)) => (rowId, predictedClass)
    }
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
    
    val trainFile = sc.textFile(System.getProperty("trainFile")).persist(spark.storage.StorageLevel.DISK_ONLY)
    val predictFile = sc.textFile(System.getProperty("predictFile")).persist(spark.storage.StorageLevel.DISK_ONLY)
    
    val trainData = trainFile.map(parseTrainData _).persist(spark.storage.StorageLevel.DISK_ONLY)//.persist(spark.storage.StorageLevel.MEMORY_AND_DISK)
    val predictData = predictFile.map(parsePredictData _).persist(spark.storage.StorageLevel.DISK_ONLY)//.persist(spark.storage.StorageLevel.MEMORY_AND_DISK)
    
    val classParams = train(trainData, sc).collect()
    val predictedClasses = predict(predictData, classParams)
    SaveRDDAsHadoopFile(predictedClasses, System.getProperty("predictionsOutputFile"))
//    predictedClasses.saveAsTextFile(System.getProperty("predictionsOutputFile"))
//    predictedClasses.saveAsTextFile(System.getProperty("predictionsOutputFile"))
  }
}
