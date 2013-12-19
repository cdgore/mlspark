package com.cdgore.test.mlspark

import org.scalatest.FunSuite
import org.scalatest.tools.ScalaTestFramework

import org.apache.log4j.{Level, Logger}

import spark.SparkContext
import spark.SparkContext._

object SparkTest extends org.scalatest.Tag("com.cdgore.test.tags.SparkTest")

trait SparkTestUtils extends FunSuite {
  var sc: SparkContext = _

  /**
   * @param name the name of the test
   * @param silenceSpark true to turn off spark logging
   */
  def sparkTest(name: String, silenceSpark : Boolean = false)(body: => Unit) {
    test(name, SparkTest){
      val origLogLevels = if (silenceSpark) SparkUtil.silenceSpark() else null
      sc = new SparkContext("local[2]", name)
      try {
        body
      }
      finally {
        sc.stop
        sc = null
        // To avoid Akka rebinding to the same port, since it doesn't unbind immediately on shutdown
        System.clearProperty("spark.master.port")
//        if (silenceSpark) Logging.setLogLevels(origLogLevels)
      }
    }
  }
}

object SparkUtil {
  def silenceSpark() {
    setLogLevels(Level.WARN, Seq("spark", "org.eclipse.jetty", "akka"))
  }

  def setLogLevels(level: org.apache.log4j.Level, loggers: TraversableOnce[String]) = {
    loggers.map{
      loggerName =>
        val logger = Logger.getLogger(loggerName)
        val prevLevel = logger.getLevel()
        logger.setLevel(level)
        loggerName -> prevLevel
    }.toMap
  }

}