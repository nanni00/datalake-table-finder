import org.apache.hadoop.conf.Configuration
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.log4j.{Level, Logger}


object CreateRawTokens {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("CreateIndex").setMaster("local[1]")
    // .set("spark.executor.memory", "100g")
    // .set("spark.driver.memory", "5g")
    val sc = new SparkContext(conf)

    print("Initial arguments: ")
    print(args.mkString(", "))
    print("\n")

    // necessary to run the jar file
    val hadoopConfig: Configuration = sc.hadoopConfiguration
    hadoopConfig.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
    hadoopConfig.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)

    // The input sets, preprocessed to remove numerical values
    val setFile = args(0)

    // The output token file.
    val rawTokensFile = args(1)

    // The tokens to skip.
    val skipTokens = Set("##################################")

    val sets = sc.textFile(setFile).map(line => line.split('|'))
      .map(line => (line(0).toInt, line.drop(1).filter(
        token => !skipTokens(token))))
    sets.flatMap { case (sid, tokens) =>
      tokens.map(token => (token, sid))
    }.map{
      case (token, sid) => s"$token $sid"
    }.saveAsTextFile(rawTokensFile)
  }
}
