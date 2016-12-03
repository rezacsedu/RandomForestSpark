package com.example.RandomForest;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import com.example.SparkSession.UtilityForSparkSession;
import scala.Tuple2;

public class JavaRandomForestRegressionExample {
	static SparkSession spark = UtilityForSparkSession.mySession();
  public static void main(String[] args) throws FileNotFoundException {
   // String datapath = "C:/Users/rezkar/Downloads/KEGG/YearPredictionMSD/YearPredictionMSD";
    String datapath = args[0];
    Dataset<Row> df = spark.read().format("libsvm").option("header", "true").load(datapath); 
    df.show();
    
    JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(spark.sparkContext(), datapath).toJavaRDD();
    // Split the data into training and test sets (89.98147% held out for training and the rest as testing)
    JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.8998147, 0.1001853});
    JavaRDD<LabeledPoint> trainingData = splits[0];
    JavaRDD<LabeledPoint> testData = splits[1];

    // Set parameters. The empty categoricalFeaturesInfo indicates all features are continuous.
    Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
    Integer numTrees = 20; // Use more in practice.
    String featureSubsetStrategy = "auto"; // Let the algorithm choose.
    String impurity = "variance";
    Integer maxDepth = 20;
    Integer maxBins = 20;
    Integer seed = 12345;
    // Train a RandomForest model.
    final RandomForestModel model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins, seed);

    // Evaluate model on test instances and compute test error
    JavaPairRDD<Double, Double> predictionAndLabel =
      testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
        @Override
        public Tuple2<Double, Double> call(LabeledPoint p) {
          return new Tuple2<>(model.predict(p.features()), p.label());
        }
      });
    
    PrintStream out = new PrintStream(new FileOutputStream("output.txt"));
    System.setOut(out);
    
    Double testMSE =
      predictionAndLabel.map(new Function<Tuple2<Double, Double>, Double>() {
        @Override
        public Double call(Tuple2<Double, Double> pl) {
          Double diff = pl._1() - pl._2();
          return diff * diff;
        }
      }).reduce(new Function2<Double, Double, Double>() {
        @Override
        public Double call(Double a, Double b) {
          return a + b;
        }
      }) / testData.count();
    
    System.out.println("Test Mean Squared Error: " + testMSE);
    //System.out.println("Learned regression forest model:\n" + model.toDebugString());
    
    // Evaluation -2: 
 // Evaluation-2: evaluate the model on test instances and compute the related performance measure statistics
    JavaRDD<Tuple2<Object, Object>> predictionAndLabels = testData.map(
  	      new Function<LabeledPoint, Tuple2<Object, Object>>() {
  			public Tuple2<Object, Object> call(LabeledPoint p) {
  	          Double prediction = model.predict(p.features());
  	          return new Tuple2<Object, Object>(prediction, p.label());
  	        }
  	      }
  	    ); 
    
    // Get evaluation metrics.
    MulticlassMetrics metrics = new MulticlassMetrics(predictionAndLabels.rdd());
    //System.out.println(metrics.confusionMatrix());
   // System.out.println(metrics.confusionMatrix());
    double precision = metrics.precision(metrics.labels()[0]);
    double recall = metrics.recall(metrics.labels()[0]);
    double f_measure = metrics.fMeasure();
    double query_label = 2001;
    double TP = metrics.truePositiveRate(query_label);
    double FP = metrics.falsePositiveRate(query_label);
    double WTP = metrics.weightedTruePositiveRate();
    double WFP =  metrics.weightedFalsePositiveRate();
    System.out.println("Precision = " + precision);
    System.out.println("Recall = " + recall);
    System.out.println("F-measure = " + f_measure);
    System.out.println("True Positive Rate = " + TP);
    System.out.println("False Positive Rate = " + FP);
    System.out.println("Weighted True Positive Rate = " + WTP);
    System.out.println("Weighted False Positive Rate = " + WFP);


    // Save and load model
    //model.save(jsc.sc(), "target/tmp/myRandomForestRegressionModel");
    //RandomForestModel sameModel = RandomForestModel.load(jsc.sc(),"target/tmp/myRandomForestRegressionModel");
    // $example off$

    spark.stop();
  }
}
