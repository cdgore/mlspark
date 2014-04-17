package com.cdgore.test.mlspark

import org.scalatest.FunSuite
import org.scalatest.matchers.ShouldMatchers
import org.scalatest.Assertions._

import com.cdgore.mlspark.GaussianNaiveBayes

import org.jblas.DoubleMatrix
import org.jblas.MatrixFunctions

import scala.util.Random

class GaussianNaiveBayesTest extends SparkTestUtils with ShouldMatchers {
  test("Applying a Gaussian function to features with zero variance should result in those features being dropped") {
    expect(2) {
      val mu = new DoubleMatrix(Array(1.4, 3.3, 4.1))
      val sigma = new DoubleMatrix(Array(0.2, 0.0, 0.4))
      val x = new DoubleMatrix(Array(1.5, 4.1, 3.1))
      
      // Expecting org.jblas.DoubleMatrix = [1.760327; NaN; 0.043821]
      // The middle variable drops out because it has zero variance
      // The training function however always adds a small number epsilon to the variance though,
      //   so if trained with this same class, there there should never be a result with exactly 0 variance
      GaussianNaiveBayes.applyGaussian(mu, sigma, x).toArray.filter(x => x >= 0).length
    }
  }
  
  sparkTest("Test for reasonable accuracy") {
    val mu11 = -1.5
    val sig11 = 2.3
    val mu12 = 3.1
    val sig12 = 5.9
    
    val mu21 = 2.5
    val sig21 = 1.5
    val mu22 = 8
    val sig22 = 2.1

    val mu31 = 5.0
    val sig31 = 1.7
    val mu32 = 0.5
    val sig32 = 2.7
    
    val classLabels = Array("A", "B", "C")
    val genParams = Map[String, Array[(Double,Double)]]("A" -> Array((mu11,sig11),(mu12,sig12)),
                        "B" -> Array((mu21,sig21),(mu22,sig22)),
                        "C" -> Array((mu31,sig31),(mu32,sig32)))
    val dataList = Random.shuffle(List.fromArray(classLabels.flatMap(x =>
      for(i <- List.range(0, 20000)) yield x).map(l => (l, new DoubleMatrix(genParams(l).map(p =>
        (p._2 * Random.nextGaussian() + p._1 )).toArray) ))))

    val trainData = sc.parallelize(dataList.slice(0, (dataList.length*0.8).toInt))
    val testData = sc.parallelize(dataList.slice((dataList.length*0.8).toInt, dataList.length))
    
    val learnedParameters = GaussianNaiveBayes.train(trainData, sc).collect()
    
    val prediction = GaussianNaiveBayes.predict(testData.map(x => ("row_id", x._1, x._2)), learnedParameters)
    
    val truePos = prediction.map(x => (if(x._1.equals(x._2)) 1 else 0, 1)).reduce((a,b) => (a._1+b._1, a._2+b._2)) match {case (s, n) => s.toDouble/n.toDouble}
    
    truePos should be >= 0.5
  }
}
