package xilodyne.udacity.intro_ml.miniproj_NB_Enron.weka;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Iterator;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader.ArffReader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;
import weka.filters.unsupervised.attribute.StringToWordVector;
import xilodyne.util.ArrayUtils;
import xilodyne.util.G;
import xilodyne.util.Logger;
import xilodyne.util.weka.WekaUtils;

public class WEKA_GNB_Enron {

	private Logger log = new Logger();
	
	private Classifier cls;
	private Instances trainingSet, testingSet;
	private AddClassification filter;
	private FilteredClassifier fc;
	private Evaluation eval;
	private double[] trainingLabels, testingLabels;
	private double[] predLabels_Test;


	public WEKA_GNB_Enron() {
		log.logln_withClassName(G.LOG_FINE, "");

		// Create a naïve bayes classifier
		cls = (Classifier) new NaiveBayes();
		filter = new AddClassification();
		fc = new FilteredClassifier();
	}

	public Instances readEnronData(String filename) {
		log.logln(G.LOG_INFO, "Reading in data from: " + filename);
		BufferedReader reader;
		Instances wekaData = null;
		try {
			reader = new BufferedReader(new FileReader(filename));
			 ArffReader arff = new ArffReader(reader);
					 wekaData = arff.getData();
					 wekaData.setClassIndex(wekaData.numAttributes() - 1);
			//wekaData = new Instances(reader);
			reader.close();
			// setting class attribute
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		wekaData.setClassIndex(wekaData.numAttributes() - 1);
		return wekaData;
	}

	/*
	 * http://stackoverflow.com/questions/28123954/how-do-i-divide-a-dataset-into-training-and-test-sets-using-weka
	 * 
	 * The way I did it (use Weka's methods), 
	 * the data is always divided so that (k-1)/k 
	 * are training set and 1/k are test set. 
	 * If you want to divide 90/10, you have to choose k=10. 
	 * And if you don't want to have 10 different splits, 
	 * use the method described above and not use a for-loop
	 */
	//split data into training and testing data
	public void generateTestData(Instances data) {
		int seed = 1;
		int folds = 10;

		log.logln(G.lI, "Generating test data from size: " + data.numInstances());
		// randomize data
		Random rand = new Random(seed);
		Instances randData = new Instances(data);
		randData.randomize(rand);

		if (randData.classAttribute().isNominal())
			randData.stratify(folds);

		// perform cross-validation
		
				trainingSet = randData.trainCV(folds, 0);
				testingSet = randData.testCV(folds, 0);
				log.logln("Sizes train/test: " + trainingSet.numInstances() +"/" + testingSet.numInstances());
		
		double[] dataLabels = WekaUtils.loadLabelsFromWekaData(data, log);
		//System.out.println("data labels: " + ArrayUtils.print1DArray(dataLabels));
				
		
		this.trainingLabels = WekaUtils.loadLabelsFromWekaData(trainingSet, log);
		//System.out.println("training labels: " + ArrayUtils.print1DArray(this.trainingLabels));
		
		this.testingLabels = WekaUtils.loadLabelsFromWekaData(testingSet, log);
		//System.out.println("testing labels: " + ArrayUtils.print1DArray(this.testingLabels));
		
	}
	
	public int getClassCountFromTrainingSet(double dClassIndex) {
		int count = 0;
		for (int index = 0; index < trainingSet.size(); index++) {
			if (trainingSet.instance(index).classValue() == dClassIndex)
				count++;
		}
		return count;

	}
	
	public Instances getTrainingSet() {
		return this.trainingSet;
	}
	public Instances getTestingSet() {
		return this.testingSet;
	}
	
	public Instances convertToTdidfVector(Instances wordData) {
		String[] options = new String[1];
	      options[0] = "-L"; 
	//      options[1] = "-R <First>";
//		String[] options = new String[]{"-L -R <1,2>"};
//		String[] options = new String[]{"-L"};
		

		StringToWordVector filter = new StringToWordVector();
		Instances newData = null;
		
		try {
			System.out.println("att indicies: " + filter.getAttributeIndices());
			System.out.println("att prefix: " + filter.getAttributeNamePrefix());

			filter.setOptions(options);
			filter.setInputFormat(wordData);
			newData = Filter.useFilter(wordData,  filter);
		} catch (Exception e) {
			e.printStackTrace();
		} 
		
	//	System.out.println("filter: " + filter.toString());
	//	System.out.println("new data: " + newData);
		return newData;
	}
	
	/*
	 			cls.buildClassifier(trainingSet);
			
			Evaluation eval = new Evaluation(trainingSet);
			 eval.evaluateModel(cls, testingSet);
			 log.logln(G.LOG_FINE, eval.toSummaryString("\nResults\n======\n", false));
			 
			predLabels_Test = new double[testingSet.numInstances()];
			for (int index = 0; index < predLabels_Test.length; index++) {
				predLabels_Test[index] = cls.classifyInstance(testingSet.instance(index));

	 */
	public void fit (Instances trainingSet) {
		try {
			cls.buildClassifier(trainingSet);
			eval = new Evaluation(trainingSet);
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
	public void testAndPredict(Instances testingSet){
		
		 try {
			eval.evaluateModel(cls, testingSet);
		} catch (Exception e) {
			e.printStackTrace();
		}
		 log.logln(G.LOG_FINE, eval.toSummaryString("\nResults\n======\n", false));
		 log.logln(String.valueOf(eval.pctCorrect()));
		//predLabels_Test = new double[testingSet.numInstances()];
		//for (int index = 0; index < predLabels_Test.length; index++) {
		//	predLabels_Test[index] = cls.classifyInstance(testingSet.instance(index));

	}
	
	
	public void fitS (Instances iTrainingSet) {
		try {
			cls.buildClassifier(iTrainingSet);
		} catch (Exception e) {
			e.printStackTrace();
		}
		//Classifier cls = new J48();
		 Evaluation eval;
		try {
			eval = new Evaluation(iTrainingSet);
			 Random rand = new Random(1);  // using seed = 1
			 int folds = 10;
			 eval.crossValidateModel(cls, iTrainingSet, folds, rand);
			 log.logln(G.lI, eval.toSummaryString());
			 log.logln(String.valueOf(eval.pctCorrect()));
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	// based on the weka example: CrossValidationSingleRun
	// populates trainingSet and testingSet
	public void crossValidate(Instances data) {

		// other options
		int seed = 1;
		int folds = 10;

		// randomize data
		Random rand = new Random(seed);
		Instances randData = new Instances(data);
		randData.randomize(rand);

		if (randData.classAttribute().isNominal())
			randData.stratify(folds);

		// perform cross-validation
		try {
			eval = new Evaluation(randData);
			for (int n = 0; n < folds; n++) {
				trainingSet = randData.trainCV(folds, n);
				testingSet = randData.testCV(folds, n);

				Classifier clsCopy = AbstractClassifier.makeCopy(cls);
				clsCopy.buildClassifier(trainingSet);
				eval.evaluateModel(clsCopy, testingSet);
			}		} catch (Exception e) {
			e.printStackTrace();
		}

		// output evaluation
		System.out.println();
		System.out.println("=== Setup ===");
		System.out.println("Classifier: " + Utils.toCommandLine(cls));
		System.out.println("Dataset: " + data.relationName());
		System.out.println("Folds: " + folds);
		System.out.println("Seed: " + seed);
		System.out.println();
		System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));

	//	System.out.println("Eval: " + eval.);
		/*
		System.out.println("training: " + trainingSet);
		
		double[] trainLabels = new double[trainingSet.numInstances()];
		System.out.println("testingSet size: " + trainingSet.numInstances());
		for (int index = 0; index < trainLabels.length; index++) {
			try {
				System.out.println(index +"\t" + clsCopy.classifyInstance(trainingSet.instance(index)) );
				trainLabels[index] = cls.classifyInstance(trainingSet.instance(index));
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		
		predLabels_Test = new double[testingSet.numInstances()];
		System.out.println("testingSet size: " + testingSet.numInstances());
		for (int index = 0; index < predLabels_Test.length; index++) {
			try {
				System.out.println(index +"\t" + cls.classifyInstance(testingSet.instance(index)) );
				predLabels_Test[index] = cls.classifyInstance(testingSet.instance(index));
			} catch (Exception e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		System.out.println("training labels: " + ArrayUtils.print1DArray(predLabels_Test));
*/
		/*for (int i = 0; i < testingSet.numInstances(); i++) {
			  double pred;
			try {
				
				pred = clsCopy.classifyInstance(testingSet.instance(i));
				//pred = fc.classifyInstance(testingSet.instance(i));
				  System.out.print("ID: " + testingSet.instance(i).value(0));
				  System.out.print(", actual: " + testingSet.classAttribute().value((int) testingSet.instance(i).classValue()));
				 System.out.println(", predicted: " + testingSet.classAttribute().value((int) pred));
			} catch (Exception e) {
				e.printStackTrace();
			}
		}*/
		
	}
	
/*	private double[] loadLabelsFromInstance(Instances tempInstance) {
		System.out.println("instance size: " + tempInstance.numInstances());
		System.out.println("# of classes: " + tempInstance.numClasses());

		double[] labels = new double[tempInstance.numInstances()];
		System.out.println("label size: " + labels.length);
		for (int index = 0; index < labels.length; index++) {
			try {
				labels[index] = tempInstance.instance(index).classValue();
			} catch (Exception e) {
				e.printStackTrace();
			}
		}
		return labels;

	}
	*/
/*
	//uses data created in cross validation routine
	public double[] Predict() {

		try {
			cls.buildClassifier(trainingSet);

			Evaluation eval = new Evaluation(trainingSet);
			eval.evaluateModel(cls, testingSet);
			// log.logln(G.LOG_FINE, eval.toSummaryString("\nResults\n======\n",
			// false));

			predLabels_Test = new double[testingSet.numInstances()];
			for (int index = 0; index < predLabels_Test.length; index++) {
				predLabels_Test[index] = cls.classifyInstance(testingSet.instance(index));
			}

		} catch (Exception e) {
			e.printStackTrace();
		}

	}
*/
	/*
	 * public void getAccuracy() { fc.buildClassifier(train); for (int i = 0; i
	 * < test.numInstances(); i++) { double pred =
	 * fc.classifyInstance(test.instance(i)); predicated =
	 * (test.classAttribute().value((int) pred)); } }
	 */
	
	public void getAccuracy() {
		try {
			fc.buildClassifier(trainingSet);
					
			for (int i = 0; i < testingSet.numInstances(); i++) {
				double[] probabilityDistribution = fc.distributionForInstance(testingSet.instance(i));
//				double pred = fc.classifyInstance(testingSet.instance(i));
//				String predicated = testingSet.classAttribute().value((int) pred);
				// Probability of the test instance beeing a "1"
				double classAtt1Prob = probabilityDistribution[0];
				// Probability of the test instance beeing a "2"
				double classAtt2Prob = probabilityDistribution[1];
				
				System.out.println(probabilityDistribution + "\t" + classAtt1Prob + "\t" + classAtt2Prob);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
//		System.out.println("cls acc:" + cls.toString());
		

	}
	public void getAccuracy(Instances data) {
		try {
			fc.buildClassifier(data);
					
			for (int i = 0; i < testingSet.numInstances(); i++) {
				double[] probabilityDistribution = fc.distributionForInstance(testingSet.instance(i));
//				double pred = fc.classifyInstance(testingSet.instance(i));
//				String predicated = testingSet.classAttribute().value((int) pred);
				// Probability of the test instance beeing a "1"
				double classAtt1Prob = probabilityDistribution[0];
				// Probability of the test instance beeing a "2"
				double classAtt2Prob = probabilityDistribution[1];
				
				System.out.println(probabilityDistribution + "\t" + classAtt1Prob + "\t" + classAtt2Prob);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		
//		System.out.println("cls acc:" + cls.toString());
		

	}

}
