import sbt._
import Keys._
import sbtassembly.Plugin._
import AssemblyKeys._

object Builds extends Build {
  // Hadoop version to build against. For example, "0.20.2", "0.20.205.0", or
  // "1.0.4" for Apache releases, or "0.20.2-cdh3u5" for Cloudera Hadoop.
  val HADOOP_VERSION = "1.0.3"
  val HADOOP_MAJOR_VERSION = "1"

  // For Hadoop 2 versions such as "2.0.0-mr1-cdh4.1.1", set the HADOOP_MAJOR_VERSION to "2"
  //val HADOOP_VERSION = "2.0.0-mr1-cdh4.1.1"
  //val HADOOP_MAJOR_VERSION = "2"
  
  val excludeJackson = ExclusionRule(organization = "org.codehaus.jackson")
  val excludeNetty = ExclusionRule(organization = "org.jboss.netty")
  val excludeAsm = ExclusionRule(organization = "asm")
  val excludeSnappy = ExclusionRule(organization = "org.xerial.snappy")
  
  lazy val logback = "ch.qos.logback" % "logback-classic" % "0.9.28" % "runtime"
  
  lazy val buildSettings = Defaults.defaultSettings ++ Seq(
    name := "MLSpark",
    organization := "com.cdgore",
    version := "0.0.1",
    scalaVersion := "2.9.3",
    // scalacOptions := Seq("-unchecked", "-optimize", "-deprecation", "-target:jvm-1.5"),
    // javacOptions := Seq("-source", "1.5", "-target", "1.5"),
    // unmanagedJars in Compile <<= baseDirectory map { base => (base / "lib" ** "*.jar").classpath },
    // retrieveManaged := true,
    // retrievePattern := "[type]s/[artifact](-[revision])(-[classifier]).[ext]",
    // transitiveClassifiers in Scope.GlobalScope := Seq("sources"),
    resolvers ++= Seq(
      "Akka Repository" at "http://repo.akka.io/releases/",
      "Spray Repository" at "http://repo.spray.cc/",
      "JBoss Repository" at "http://repository.jboss.org/nexus/content/repositories/releases/",
      "Cloudera Repository" at "https://repository.cloudera.com/artifactory/cloudera-repos/"),
    libraryDependencies ++= Seq(
      // "org.spark-project" %% "spark-core" % "0.7.3" excludeAll(excludeNetty),
      "org.spark-project" %% "spark-core" % "0.7.3",
      "org.scalanlp" % "jblas" % "1.2.1",
      "org.apache.mahout" % "mahout-core" % "0.8",
      "org.apache.mahout" % "mahout-math" % "0.8",
      // "org.jboss.netty" % "netty" % "3.2.4.Final",
      "com.typesafe.akka" % "akka-actor" % "2.0.5" excludeAll(excludeNetty),
      "com.typesafe.akka" % "akka-remote" % "2.0.5" excludeAll(excludeNetty),
      "com.typesafe.akka" % "akka-slf4j" % "2.0.5" excludeAll(excludeNetty)
      // "org.eclipse.jetty" % "jetty-server" % "7.6.8.v20121106",
      // "org.scalatest" %% "scalatest" % "1.9.1" % "test",
      // "org.scalacheck" %% "scalacheck" % "1.10.0" % "test",
      // "com.novocode" % "junit-interface" % "0.9" % "test",
      // "org.easymock" % "easymock" % "3.1" % "test"
    )
  )
  
  def oozieSettings = Seq(
    excludedJars in assembly <<= (fullClasspath in assembly) map { cp => 
      cp filter {_.data.getName == "netty-3.5.4.Final.jar"}
    }
  )
  
  lazy val app = Project("mlspark", file("."),
    // settings = buildSettings ++ Seq( 
      // settings = buildSettings ++ oozieSettings ++ assemblySettings) settings(
      settings = buildSettings ++ assemblySettings) settings(
      mergeStrategy in assembly <<= (mergeStrategy in assembly) { (old) =>
        {
          case PathList("javax", "servlet", xs @ _*)         => MergeStrategy.first
          case PathList(ps @ _*) if ps.last endsWith ".html" => MergeStrategy.first
          case PathList("org", "apache", xs @ _*) => MergeStrategy.first
          case "application.conf" => MergeStrategy.concat
          case "unwanted.txt"     => MergeStrategy.discard
          case m if m.toLowerCase.endsWith("manifest.mf") => MergeStrategy.discard
          case m if m.toLowerCase.matches("meta-inf.*\\.sf$") => MergeStrategy.discard
          case "reference.conf" => MergeStrategy.concat
          case _ => MergeStrategy.first
          // case x => old(x)
        }
      }
    )
  // )
}
