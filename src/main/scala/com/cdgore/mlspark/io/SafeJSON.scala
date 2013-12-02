package com.retentionscience.rsml.io

import java.util.Random
import spark.SparkContext
import spark.util.Vector
import spark.SparkContext._
import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet
import scala.util.parsing.json._

import java.io._

class SafeJSON extends Parser { 

  private def unRaw (in : Any) : Any = in match { 
    case JSONObject(obj) => obj.map({ case (k,v) => (k,unRaw(v))}).toList 
    case JSONArray(list) => list.map(unRaw) 
    case x => x 
  } 

  def parseRaw(input : String) : Option[JSONType] = 
    phrase(root)(new lexical.Scanner(input)) match { 
      case Success(result, _) => Some(result) 
      case _ => None 
    } 

  def parseFull(input: String): Option[Any] = 
    parseRaw(input) match { 
      case Some(data) => Some(resolveType(data)) 
      case None => None 
    } 

  def resolveType(input: Any): Any = input match { 
    case JSONObject(data) => data.transform { 
      case (k,v) => resolveType(v) 
    } 
    case JSONArray(data) => data.map(resolveType) 
    case x => x 
  } 

  def perThreadNumberParser_=(f : NumericParser) { numberParser.set(f) } 
  def perThreadNumberParser : NumericParser = numberParser.get() 
} 

object SafeJSON { 
  val parser = new ThreadLocal[SafeJSON] { 
    override def initialValue = new SafeJSON 
  } 

  def parseRaw(input: String): Option[JSONType] = parser.get.parseRaw(input) 
  def parseFull(input: String): Option[Any] = parser.get.parseFull(input) 

  def resolveType(input: Any): Any = parser.get.resolveType(input) 

  // additional methods can be similarly proxied here... 
} 

