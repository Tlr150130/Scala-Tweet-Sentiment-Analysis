// Databricks notebook source
// libraries
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.functions.udf
import java.util.regex.Pattern

// library for metrics
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils

// COMMAND ----------

// Functions and values:
// 1. Remove Puctuations
val removePunctuationAndSpecialChar = udf {
  (text: String) =>
    val regex = "[\\.\\,\\:\\-\\!\\?\\n\\t,\\%\\#\\*\\|\\=\\(\\)\\\"\\>\\<\\/\\'\\`\\&\\{\\}\\;\\+\\-\\[\\]\\_\\@]"
    val pattern = Pattern.compile(regex)
    val matcher = pattern.matcher(text)
  
    // Remove all matches, split at whitespace )repeated whitspace is allowed) then join again
    val cleanedText = matcher.replaceAll(" ").split("[ ]+").mkString(" ")
    cleanedText
};
// 2. All letters are lowercase
val toLowerCase = udf {
  (text: String) => { text.toLowerCase }
};
// 3. Load Data and clean
def loadData(filename: String): DataFrame = {
  val data = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(filename)
    .select("airline_sentiment", "text")
    .toDF("sentiment", "text")
    .na.drop()
  val result: DataFrame = data.withColumn("textClean", removePunctuationAndSpecialChar(toLowerCase(data.col("text"))))
  return result
}
// 4. Function to Tokenize Data
val tokenizer = new Tokenizer()
  .setInputCol("textClean")
  .setOutputCol("words")
// 5. Function to remove stopwords
val remover = new StopWordsRemover()
  .setInputCol("words")
  .setOutputCol("filtered")
// 6. Function to hash features into a single feature array column
val hashingTF = new HashingTF()
  .setInputCol(remover.getOutputCol)
  .setOutputCol("features")
// 7. Function to turn categorical variables into levels
val indexer = new StringIndexer()
  .setInputCol("sentiment")
  .setOutputCol("label")
// 8. Declare Logistic Regression Model
val lr = new LogisticRegression() 
  .setMaxIter(30)
  .setElasticNetParam(0.9)
// 9. Create a Pipeline that goes through each stage
val pipeline = new Pipeline()
  .setStages(Array(tokenizer, remover, hashingTF, indexer, lr))
// 10. parameter grid
val paramGrid = new ParamGridBuilder()
  .addGrid(hashingTF.numFeatures, Array(10, 100))
  .addGrid(lr.regParam, Array(10, 1, 0.1, 0.01)) // include other values
  .build()
// 11. 10-fold Cross-Validation to get best set of parameters
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(new BinaryClassificationEvaluator)
  .setEstimatorParamMaps(paramGrid)
  .setNumFolds(10)  // Use 3+ in practice
  .setParallelism(2) // there are only 2 parameters being tested

// COMMAND ----------

// Filename Variable
val filename = "/FileStore/tables/Tweets.csv"

// Call pipeline
var data: DataFrame = loadData(filename)
val Array(training, test) = data.randomSplit(Array(0.8, 0.2), seed = 1) // split data
val cvModel = cv.fit(training)
val results = cvModel.transform(test)
  .select("label", "prediction")

// COMMAND ----------

// turn dataframe into rdd for metric evaluation
val resultRDD = results.as[(Double,Double)].rdd

// Instantiate metrics object
val metrics = new MulticlassMetrics(resultRDD)

// Confusion matrix
println("Confusion matrix:")
println(metrics.confusionMatrix)

// Accuracy
val accuracy = metrics.accuracy
println("Summary Statistics")
println(s"Accuracy = $accuracy")

// Precision by label
val labels = metrics.labels
labels.foreach { l =>
  println(s"Precision($l) = " + metrics.precision(l))
}
// Recall by label
labels.foreach { l =>
  println(s"Recall($l) = " + metrics.recall(l))
}
// F-measure by label
labels.foreach { l =>
  println(s"F1-Score($l) = " + metrics.fMeasure(l))
}
