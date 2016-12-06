package com.example.RandomForest;

import java.util.HashMap;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.RandomForest;
import org.apache.spark.mllib.tree.model.RandomForestModel;
import org.apache.spark.mllib.util.MLUtils;
import org.apache.spark.sql.SparkSession;
import com.example.SparkSession.UtilityForSparkSession;
import scala.Tuple2;

public class JavaRandomForestClassificationExample {
	static SparkSession spark = UtilityForSparkSession.mySession();
	
  public static void main(String[] args) {
    // Load and parse the data file.
    String datapath = "input/Letterdata_libsvm.data";
    JavaRDD<LabeledPoint> data = MLUtils.loadLibSVMFile(spark.sparkContext(), datapath).toJavaRDD();
    // Split the data into training and test sets (30% held out for testing)
    JavaRDD<LabeledPoint>[] splits = data.randomSplit(new double[]{0.7, 0.3}, 12345);
    JavaRDD<LabeledPoint> trainingData = splits[0];
    JavaRDD<LabeledPoint> testData = splits[1];

    // Train a RandomForest model. Empty categoricalFeaturesInfo indicates all features are continuous.
    Integer numClasses = 26;
    HashMap<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
    Integer numTrees = 10; // Use more in practice.
    String featureSubsetStrategy = "auto"; // Let the algorithm choose feature subset strategy. 
    String impurity = "gini";
    Integer maxDepth = 30;
    Integer maxBins = 40;
    Integer seed = 12345;

    final RandomForestModel model = RandomForest.trainClassifier(trainingData, numClasses,
      categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins,
      seed);

    // Evaluation-1: evaluate the model on test instances and compute test error
    JavaPairRDD<Double, Double> predictionAndLabel =
      testData.mapToPair(new PairFunction<LabeledPoint, Double, Double>() {
        @Override
        public Tuple2<Double, Double> call(LabeledPoint p) {
          return new Tuple2<>(model.predict(p.features()), p.label());
        }
      });
    
    Double testErr =
      1.0 * predictionAndLabel.filter(new Function<Tuple2<Double, Double>, Boolean>() {
        @Override
        public Boolean call(Tuple2<Double, Double> pl) {
          return !pl._1().equals(pl._2());
        }
      }).count() / testData.count();
    System.out.println("Test Error: " + testErr);
    System.out.println("Learned classification forest model:\n" + model.toDebugString());
    
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
    System.out.println(metrics.confusionMatrix());
    System.out.println(metrics.confusionMatrix());
    double precision = metrics.precision(metrics.labels()[0]);
    double recall = metrics.recall(metrics.labels()[0]);
    double f_measure = metrics.fMeasure();
    double query_label = 8.0;
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
    //model.save(spark.sparkContext(), "target/tmp/myRandomForestClassificationModel");
    //RandomForestModel sameModel = RandomForestModel.load(spark.sparkContext(), "target/tmp/myRandomForestClassificationModel");

    spark.stop();
  }
}
