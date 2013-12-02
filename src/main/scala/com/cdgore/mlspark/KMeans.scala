/**
 * K means clustering with kernels
 * @author cdgore
 * 2013-06-01
 * Refactored into rsml 2013-09-26
 */
package com.cdgore.mlspark

import java.io._
import java.util.Random
import java.util.Properties

import spark.SparkContext
import spark.SparkContext._
import spark.util.Vector

import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.collection.JavaConverters._

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.hadoop.io.IntWritable
import org.apache.hadoop.io.DoubleWritable
import org.apache.hadoop.io.Writable
import org.apache.hadoop.io.Text

import org.apache.hadoop.mapreduce.lib.input.SequenceFileInputFormat;

//import org.apache.mahout.math.Vector
import org.apache.mahout.math.VectorWritable
import org.apache.mahout.math.DenseVector

import org.jblas.DoubleMatrix

/**
 * @author cdgore
 *
 */
//object ClusterCosRBF extends app {
object KMeans extends Serializable {
  //  val R = 1000     // Scaling factor
  //  val rand = new Random(42)

  def VectorWritableToJBlasDoubleMatrix(line: VectorWritable): DoubleMatrix = {
    return new DoubleMatrix(line.get.all.asScala.map(_.get).toArray)
  }
  
  def JBlasDoubleMatrixToVectorWritable(v: DoubleMatrix): VectorWritable = {
    return new VectorWritable(new DenseVector(v.toArray()))
  }
  
  def parseStringToStringDoubleMatrix(line: String, classIndex: Int, vectorStartIndex: Int): (String, DoubleMatrix) = {
    return (line.split('\t')(classIndex), new DoubleMatrix(line.toString().split('\t').slice(vectorStartIndex, line.length).map(_.toDouble)))
  }
  
  def parseVector(key: IntWritable, line: Text): Vector = {
    return new Vector(line.toString().split('\t').map(_.toDouble))
  }
  
  def parseTextToVector(line: Text): Vector = {
    return new Vector(line.toString().split('\t').map(_.toDouble))
  }
  
  def parseTextToDoubleMatrix(line: Text): DoubleMatrix = {
    return new DoubleMatrix(line.toString().split('\t').map(_.toDouble))
  }

  def parseIntDoubleMatrix(key: IntWritable, line: Text): (Int, DoubleMatrix) = {
    return (key.get(), new DoubleMatrix(line.toString().split('\t').map(_.toDouble)))
  }
  
  def parseTextDoubleMatrix(key: Text, line: Text): (Text, DoubleMatrix) = {
    return (key, new DoubleMatrix(line.toString().split('\t').map(_.toDouble)))
  }
  
  def parseTrainData(line: String): (String, DoubleMatrix) = {
    return (line.split("\t")(2), new DoubleMatrix(line.split("\t").slice(3, line.length).map(_.toDouble)))
  }
  
  //  def assignCentroid(dPoint: List[Double], kPoints: List[List[Double]]): Int = {
//  def assignCentroid(dPoint: Vector, kPoints: Array[Vector], distMeasure: (DoubleMatrix, DoubleMatrix) => Double): Int = {
//    var minDist = Double.PositiveInfinity
//    var assignedCluster = Vector(0.0)
//    var assignedClusterInd = 0
//
//    kPoints.zipWithIndex.foreach {
//      case (k, kIndex) =>
//        val cosD = distMeasure(dPoint, k)
//        if (cosD < minDist) {
//          minDist = cosD
//          assignedCluster = k
//          assignedClusterInd = kIndex
//        }
//    }
//
//    return assignedClusterInd
//  }
//  
  def assignCentroid(dPoint: DoubleMatrix, kPoints: Array[DoubleMatrix], distMeasure: (DoubleMatrix, DoubleMatrix) => Double): Int = {
    var minDist = Double.PositiveInfinity
    var assignedCluster = new DoubleMatrix(Array(0.0))
    var assignedClusterInd = 0

    kPoints.zipWithIndex.foreach {
      case (k, kIndex) =>
        val cosD = distMeasure(dPoint, k)
        if (cosD < minDist) {
          minDist = cosD
          assignedCluster = k
          assignedClusterInd = kIndex
        }
    }

    return assignedClusterInd
  }

//  def encodeRBFs(dPoint: Vector, kPoints: Array[Vector]): Int = {
//    var minDist = Double.PositiveInfinity
//    var assignedCluster = Vector(0.0)
//    var assignedClusterInd = 0
//
//    kPoints.zipWithIndex.foreach {
//      case (k, kIndex) =>
//        val cosD = cosDist(dPoint, k)
//        if (cosD < minDist) {
//          minDist = cosD
//          assignedCluster = k
//          assignedClusterInd = kIndex
//        }
//    }
//
//    return assignedClusterInd
//  }
  
  def norm(a: Vector): Double = {
    return math.sqrt(a.dot(a))
  }
    //  def cosDist(a: List[Double], b: List[Double]): Double = {
  def cosDist(a: Vector, b: Vector): Double = {
    return 1.0 - a.dot(b) / (math.sqrt(a.dot(a)) * math.sqrt(b.dot(b)))
  }
  
  def cosDist(a: DoubleMatrix, b: DoubleMatrix): Double = {
    return 1.0 - a.dot(b) / (math.sqrt(a.dot(a)) * math.sqrt(b.dot(b)))
  }
  
  def eucDist(a: DoubleMatrix, b: DoubleMatrix): Double = {
    return math.sqrt(a.sub(b).dot(a.sub(b)))
  }
  
  def getRBFValue(mu: DoubleMatrix, x: DoubleMatrix, sig: Double, distMeasure: (DoubleMatrix, DoubleMatrix) => Double): Double = {
    return math.exp(-(math.pow(distMeasure(mu, x),2))/(2*math.pow(sig,2)))
  }

  def encodeRBFs(dPoint: DoubleMatrix, modelParams: Array[(Int, (DoubleMatrix, Double))], distMeasure: (DoubleMatrix, DoubleMatrix) => Double): DoubleMatrix = {
    return new DoubleMatrix(modelParams.map {
      case (clusterID, (mu, sigma)) => getRBFValue(mu, dPoint, sigma, distMeasure)
    })
  }
  
  def closestPoint(p: Vector, centers: Array[Vector]): Int = {
    var index = 0
    var bestIndex = 0
    var closest = Double.PositiveInfinity

    for (i <- 0 until centers.length) {
      val tempDist = p.squaredDist(centers(i))
      if (tempDist < closest) {
        closest = tempDist
        bestIndex = i
      }
    }
    return bestIndex
  }

//  def generateClusters(data: spark.RDD[Vector], K: Int, maxNumIter: Int, outputFile: String) {
  def generateClusters(data: spark.RDD[DoubleMatrix], distMeasure: (DoubleMatrix, DoubleMatrix) => Double,  K: Int, maxNumIter: Int, outputFile: String) {
    var clusterMeans = data.takeSample(false, K, 42).toArray
    var clusterSigmas = Array[Double]()
    var previousCentroids = Array[DoubleMatrix]()
    var numIter = 0
    var continueIterating = true
    while (continueIterating) {
      numIter += 1
      println("Running cluster iteration: " + numIter)
//      println("Cluster means: " + clusterMeans.toString())
      previousCentroids = clusterMeans.clone()
      //      var closest = data.map (p => (closestPoint(p, clusterMeans), (p, 1)))
      val assignedClusters = data.map(
        p => (assignCentroid(p, clusterMeans, distMeasure), (p, 1)))

      // Recalculate the mean and standard deviation for each cluster 
      val newClusterMeans = assignedClusters.reduceByKey {
        case ((x1, y1), (x2, y2)) => (x1.addi(x2), y1 +y2)
      }.map {
        case (x, (y, n)) => (x, (y.div(n)).div(y.div(n).norm2()))
      }
      
      val newClusterPairs = assignedClusters.join(newClusterMeans)
      
//      val newClusterSigmas = newClusterPairs.map {
//        case (k, ((x, c), u)) => (k, (math.pow(distMeasure(x, u), 2), c))
//      }.reduceByKey {
//        case ((x1, y1), (x2, y2)) => (x1 + x2, y1 + y2)
//      }.map {
//        case (x, (y, n)) => (x, math.sqrt(y / n))
//      }
      
      val mus = newClusterMeans.collect()
//      val sigmas = newClusterSigmas.collect()
      
      for (m <- mus) {
        clusterMeans(m._1) = m._2
      }
      
      // If the conditions are met to stop iterating, dump the learned cluster parameters to files
      if ((previousCentroids, clusterMeans).zipped.map { case (x, y) => x.sub(y) }.reduceLeft((x, y) => x.addi(y)).sum == 0.0) {
        println("FINISHED ITERATING: Clusters converged")
        continueIterating = false
      }
      if (numIter >= maxNumIter) {
        println("FINISHED ITERATING: Maximum number of iterations reached")
        continueIterating = false
      }
      if (continueIterating == false) {
        val newClusterSigmas = newClusterPairs.map {
          case (k, ((x, c), u)) => (k, (math.pow(distMeasure(x, u), 2), c))
        }.reduceByKey {
          case ((x1, y1), (x2, y2)) => (x1 + x2, y1 + y2)
        }.map {
          case (x, (y, n)) => (x, math.sqrt(y / n))
        }
        val outputClusterMeansVectorWritable = newClusterMeans.map {
          case (x, y) => (new IntWritable(x), JBlasDoubleMatrixToVectorWritable(y))
        }
        outputClusterMeansVectorWritable.saveAsSequenceFile(outputFile + "_means")
        println("Wrote " + outputFile + "_means")
        newClusterSigmas.saveAsSequenceFile(outputFile + "_sigmas")
        println("Wrote " + outputFile + "_sigmas")
      }
    }
  }
  
  def train(sc: SparkContext) {
    val K = System.getProperty("K").toInt// Number of clusters
    val maxNumIter = System.getProperty("maxNumIter").toInt// Maximum number of iterations
    
    val inputFile = System.getProperty("inputFile")
    val outputFile = System.getProperty("parametersOutputFile")
    
    val classIndex = System.getProperty("classIndex") match {
      case x:String => x.toInt
      case _ => 0
    }
    
    val vectorStartIndex = System.getProperty("vectorStartIndex") match {
      case x:String => x.toInt
      case _ => 1
    }
    
//    val lines = sc.sequenceFile[IntWritable, VectorWritable](inputFile)//.persist(spark.storage.StorageLevel.DISK_ONLY)
//    val data = lines.map(x => VectorWritableToJBlasDoubleMatrix(x._2))
    
    val data = System.getProperty("inputFormat") match {
      case "seq" => sc.sequenceFile[IntWritable, VectorWritable](inputFile).map(x => VectorWritableToJBlasDoubleMatrix(x._2))
      case "tsv" => sc.textFile(inputFile).map(x => parseStringToStringDoubleMatrix(x, classIndex, vectorStartIndex)).map{case(k,v)=>v}
    }
    
    System.getProperty("distMeasure") match {
      case "cosine" => generateClusters(data, cosDist, K, maxNumIter, outputFile)
      case "euclidean" => generateClusters(data, eucDist, K, maxNumIter, outputFile)
    }
  }
  
  def encode(sc: SparkContext) {
    val inputFile = System.getProperty("inputFile")
    val outputFile = System.getProperty("outputFile")
    val means = sc.sequenceFile[IntWritable, VectorWritable](System.getProperty("meansFile")).map {
      case(k,v) => (k.get, VectorWritableToJBlasDoubleMatrix(v))
    }
    val sigmas = sc.sequenceFile[IntWritable, DoubleWritable](System.getProperty("sigmasFile")).map {
      case(k,v) => (k.get, v.get)
    }
    val classIndex = System.getProperty("classIndex") match {
      case x:String => x.toInt
      case _ => 0
    }
    val vectorStartIndex = System.getProperty("vectorStartIndex") match {
      case x:String => x.toInt
      case _ => 1
    }
    val data = System.getProperty("inputFormat") match {
      case "seq" => sc.sequenceFile[IntWritable, VectorWritable](inputFile).map{
        case(k,v) => (k.toString, VectorWritableToJBlasDoubleMatrix(v))}
      case "tsv" => sc.textFile(System.getProperty("inputFile")).map(
          x => parseStringToStringDoubleMatrix(x, classIndex, vectorStartIndex))
    }
    val modelParams = means.join(sigmas).sortByKey().collect()
    val finalRows = System.getProperty("distMeasure") match {
      case "cosine" => data.map{case (k,v) => (k,encodeRBFs(v, modelParams, cosDist).toArray.map(_ toString).reduceLeft((a: String, b: String) => a + '\t' + b))}
      case "euclidean" => data.map{case (k,v) => (k,encodeRBFs(v, modelParams, eucDist).toArray.map(_ toString).reduceLeft((a: String, b: String) => a + '\t' + b))}
    }
    System.getProperty("outputFormat") match {
      case "seq" => finalRows.map{case (k,v) => (new Text(k), new Text(v))}.saveAsSequenceFile(outputFile)
      case "tsv" => finalRows.saveAsTextFile(outputFile)
      case _ => finalRows.saveAsTextFile(outputFile)
    }
  }
  
//  override def main(args: Array[String]) {
  def main(args: Array[String]) {
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
    val sc = new SparkContext(master,"Clustering with Kernels", System.getProperty("spark.home"),
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

    System.getProperty("encode") match {
      case "true" => (System.getProperty("inputFile") == null || System.getProperty("outputFile") == null ||
        System.getProperty("meansFile") == null || System.getProperty("sigmasFile") == null ||
        System.getProperty("inputFormat") == null || System.getProperty("distMeasure") == null) match {
          case true => throw new IOException("ERROR: If property 'encode' is set to 'true', properties 'inputFile', 'outputFile', " + 
              "'meansFile', 'sigmasFile', 'inputFormat' (seq or tsv), and 'distMeasure' (cosine or euclidean) must also be set")
          case _ => encode(sc)
        }
      case _ => (System.getProperty("inputFile") == null || System.getProperty("parametersOutputFile") == null ||
        System.getProperty("maxNumIter") == null || System.getProperty("K") == null || System.getProperty("inputFormat") == null || 
        System.getProperty("distMeasure") == null) match {
          case true => throw new IOException("ERROR: Must specify properties 'inputFile', 'parametersOutputFile', 'K', 'maxNumIter', " + 
              "'inputFormat' (seq or tsv), and 'distMeasure' (cosine or euclidean)")
          case _ => train(sc)
        }
    }
    
    //    System.getProperties().list(System.out)

//    if (System.getProperty("inputFile") == null || System.getProperty("parametersOutputFile") == null ||
//      System.getProperty("maxNumIter") == null || System.getProperty("K") == null || System.getProperty("inputFormat") == null ||
//      System.getProperty("distMeasure") == null)
//      throw new IOException("ERROR: Must specify properties 'inputFile', 'parametersOutputFile', 'K', 'maxNumIter', 
//    " + "'inputFormat' (seq or tsv), and 'distMeasure' (cosine or euclidean)")
//    System.getProperties().list(System.out)
    
//    val K = System.getProperty("K").toInt// Number of clusters
//    val maxNumIter = System.getProperty("maxNumIter").toInt// Maximum number of iterations
//    
//    val inputFile = System.getProperty("inputFile")
//    val outputFile = System.getProperty("parametersOutputFile")
//    
////    val lines = sc.sequenceFile[IntWritable, VectorWritable](inputFile)//.persist(spark.storage.StorageLevel.DISK_ONLY)
////    val data = lines.map(x => VectorWritableToJBlasDoubleMatrix(x._2))
//    
//    val data = System.getProperty("inputFormat") match {
//      case "seq" => sc.sequenceFile[IntWritable, VectorWritable](inputFile).map(x => VectorWritableToJBlasDoubleMatrix(x._2))
//      case "tsv" => sc.textFile(inputFile).map(parseTrainData _).map{case(k,v)=>v}
//    }
//    
//    System.getProperty("distMeasure") match {
//      case "cosine" => generateClusters(data, cosDist, K, maxNumIter, outputFile)
//      case "euclidean" => generateClusters(data, eucDist, K, maxNumIter, outputFile)
//    }
//    
//    System.getProperty("encode") match {
//      case "true" => encode()
//    }
//    
//    generateClusters(data, distMeasure, K, maxNumIter, outputFile)

    System.exit(0)
  }
}
