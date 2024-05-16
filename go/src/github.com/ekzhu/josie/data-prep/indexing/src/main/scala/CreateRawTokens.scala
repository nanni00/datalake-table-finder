import org.apache.spark.{SparkConf, SparkContext}

object CreateRawTokens {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("CreateIndex").setMaster("local[1]")
    //  .set("spark.executor.memory", "100g")
    //  .set("spark.driver.memory", "5g")
    val sc = new SparkContext(conf)

    // The input sets, preprocessed to remove numerical values
     val setFile = args(0)

    // The output token file.
     val rawTokensFile = args(1)

    // The tokens to skip.
     val skipTokens = Set("acssf")

    // For web table
//    val setFile = "wdc_webtables_2015_english_relational.set"
//    val rawTokensFile = "wdc_webtables_2015_english_relational.raw-token"
//    val skipTokens = Set("-", "--", "n/a", "total", "$", ":", "*", "+", "�", "@", "†", "▼")

    val sets = sc.textFile(setFile).map(line => line.split(","))
      .map(line => (line(0).toInt, line.drop(1).filter(
        token => !skipTokens(token))))
    sets.flatMap { case (sid, tokens) =>
      tokens.map(token => (token, sid))
    }.map{
      case (token, sid) => s"$token $sid"
    }.saveAsTextFile(rawTokensFile)
  }
}
