package machineLearning

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.LinearRegression

object HousePriceUsingLingerRegression {
  def main(args:Array[String]){
    val conf=new SparkConf().setAppName("Regression Model")
                            .setMaster("local[*]")
    val sc=new SparkContext(conf)
    val sqlContext=new SQLContext(sc)
    import sqlContext.implicits._
    val data=sqlContext.read.option("inferSchema","true").option("header", true).csv("C:\\Users\\AJAIN\\Desktop\\operator_data\\Clean-USA-Housing.csv")
    data.printSchema()
    val df=data.select($"Price".as("label"), $"Avg Area Income",$"Avg Area House Age",$"Avg Area Number of Rooms",$"Area Population")
    df.show()
    df.printSchema()
    val assembler=new VectorAssembler().setInputCols(Array("Avg Area Income","Avg Area House Age","Avg Area Number of Rooms","Area Population"))
                                  .setOutputCol("features")
    val output=assembler.transform(df).select($"label",$"features")
    val lr=new LinearRegression()
    //trained the model using traing dataset
    val lrModel=lr.fit(output)
    val modelSummary=lrModel.summary
    //Slop of regression Line
    println("R2::: "+modelSummary.r2)
    println("RMSE::: "+modelSummary.predictions)
    //Predicted value and actual values
    modelSummary.predictions.show()
    modelSummary.residuals.show()
  }
}