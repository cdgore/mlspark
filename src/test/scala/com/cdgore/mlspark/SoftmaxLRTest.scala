package com.cdgore.test.mlspark

import org.scalatest.FunSuite
import org.scalatest.matchers.ShouldMatchers

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._

import com.cdgore.mlspark.SoftmaxLR

import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions

import scala.util.Random

class SoftMaxLRTest extends SparkTestUtils with ShouldMatchers {
  sparkTest("L1 Regularization with Clipping Test") {
    val data = sc.parallelize(Array("A", "B", "C").flatMap(x => for(i <- List.range(1, 301)) yield x).map(l => (l, new DoubleMatrix((1 to 500).map(x => (Random.nextGaussian() + l.getBytes.map(x => (  (x.asInstanceOf[Double] % 3)+1  )).sum)/3.0   ).toArray) )))
    val miniBatchTrainingPercentage = 0.7
    val maxIterations = 100
    val regUpdate = SoftmaxLR.l1ClippingUpdate _
    val regLambda = 0.05
    val learningRateAlpha = 0.04
    val lossFileOut = null
    
    val res = SoftmaxLR.trainLR(sc, data, learningRateAlpha, regLambda, regUpdate, miniBatchTrainingPercentage, maxIterations, lossFileOut)

    res.first._2.toArray.filter(x => x==0).length should be > 0
    res.first._2.toArray.filter(x => x==0).length should be < res.first._2.toArray.length
  }
  
  sparkTest("L2 Regularization Test") {
    val data = sc.parallelize(Array("A", "B", "C").flatMap(x => for(i <- List.range(1, 301)) yield x).map(l => (l, new DoubleMatrix((1 to 500).map(x => (Random.nextGaussian() + l.getBytes.map(x => (  (x.asInstanceOf[Double] % 3)+1  )).sum)/3.0   ).toArray) )))
    val miniBatchTrainingPercentage = 0.7
    val maxIterations = 100
    val regUpdate = SoftmaxLR.l2Update _
    val regLambda = 0.05
    val learningRateAlpha = 0.04
    val lossFileOut = null
    
    val res = SoftmaxLR.trainLR(sc, data, learningRateAlpha, regLambda, regUpdate, miniBatchTrainingPercentage, maxIterations, lossFileOut)

    res.first._2.toArray.filter(x => x==0).length should be (0)
  }
  
  sparkTest("Regression without Regularization Test") {
    val data = sc.parallelize(Array("A", "B", "C").flatMap(x => for(i <- List.range(1, 301)) yield x).map(l => (l, new DoubleMatrix((1 to 500).map(x => (Random.nextGaussian() + l.getBytes.map(x => (  (x.asInstanceOf[Double] % 3)+1  )).sum)/3.0   ).toArray) )))
    val miniBatchTrainingPercentage = 0.7
    val maxIterations = 100
    val regUpdate = SoftmaxLR.simpleUpdate _
    val regLambda = 0.05
    val learningRateAlpha = 0.04
    val lossFileOut = null
    
    val res = SoftmaxLR.trainLR(sc, data, learningRateAlpha, regLambda, regUpdate, miniBatchTrainingPercentage, maxIterations, lossFileOut)

    res.first._2.toArray.filter(x => x==0).length should be (0)
  }

//  test("non-spark code") {
//    val x = 17
//    val y = 3
//    SoftMaxLRTest.plus(x,y) should be (20)
//  }
}