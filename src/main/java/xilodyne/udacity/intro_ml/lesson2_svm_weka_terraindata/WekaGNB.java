package xilodyne.udacity.intro_ml.lesson2_svm_weka_terraindata;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Random;

import mikera.arrayz.INDArray;
import mikera.arrayz.NDArray;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.OptionHandler;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;
import xilodyne.numpy.J2NumPY;
import xilodyne.util.ArrayUtils;
import xilodyne.util.G;
import xilodyne.util.Logger;
//import xilodyne.util.WekaUtils;
import xilodyne.util.weka.WekaUtils;

public class WekaGNB {
	
	private Logger log = new Logger();
	
	private static final String sFast="0";
	private static final String sSlow="1";
	private static final int iFast=0;
	private static final int iSlow=1;

	private final int folds=10;
	private final int seed=1;
	// Declare two numeric attributes
	private Attribute Attribute1;
	private Attribute Attribute2;

	// Declare the class attribute along with its values
	private ArrayList<String> alClassVal;
	private Attribute ClassAttribute; 
	private ArrayList<Attribute> alWekaAttributes;
	private Classifier cls;
	private Instances trainingSet, testingSet, boundarySet;
	private AddClassification filter;
    private FilteredClassifier fc;
    private double[][] predLabels_DescBound;
    private double[] predLabels_Test;
	
	double x_min = 0.0, x_max = 1.0;
	double y_min = 0.0, y_max = 1.0;
	double h = 0.01;
	
	public WekaGNB() {
		// Declare two numeric attributes
		 Attribute1 = new Attribute("Xcoord");
		 Attribute2 = new Attribute("Ycoord");
	 
		// Declare the class attribute along with its values
		alClassVal = new ArrayList<String>();
		alClassVal.add(WekaClassifiers.sFast);
		alClassVal.add(WekaClassifiers.sSlow);

		ClassAttribute = new Attribute("classSpeed", alClassVal); 

		// Declare the feature vector
		 alWekaAttributes = new ArrayList<Attribute>();
		 alWekaAttributes.add(Attribute1);
		 alWekaAttributes.add(Attribute2);
		 alWekaAttributes.add(ClassAttribute);
		 
		 // Create a naïve bayes classifier
		 cls = (Classifier)new NaiveBayes();
		 filter = new AddClassification();
		 fc = new FilteredClassifier();
	}
	
	public void loadTrainingData(NDArray features_train, double[] labels_train) {
		 
		 // Create an empty training set
		 trainingSet = new Instances("Rel", alWekaAttributes, 10);
		 // Set class index
		 trainingSet.setClassIndex(2);
		 
		 System.out.println("Load training data.");
		 this.loadAttributes(trainingSet, features_train,  labels_train);

		 try {
			cls.buildClassifier(trainingSet);
		 // Create an empty training set
		 Instances isTestingSet = new Instances("Rel", alWekaAttributes, 10);
		 // Set class index
		 isTestingSet.setClassIndex(2);

		} catch (Exception e) {
			e.printStackTrace();
		}

		 
	}
	
	
	public void loadTrainingAndPredict(NDArray features_train, NDArray features_test, 
			double[] labels_train, double[] dLabels_test) {
		 
		 // Create an empty training set
		 trainingSet = new Instances("terrain", alWekaAttributes, 10);
		 testingSet = new Instances("testing", alWekaAttributes, 10);
		 boundarySet = new Instances("boundary", alWekaAttributes, 10);
		 // Set class index
		 trainingSet.setClassIndex(2);
		 testingSet.setClassIndex(2);
		 boundarySet.setClassIndex(2);
		 
		 this.loadAttributes(trainingSet, features_train,  labels_train);
		 this.loadAttributes(testingSet, features_test,  dLabels_test);

		 double[][] XX = J2NumPY.meshgrid_getXX(J2NumPY.arange(x_min, x_max, h), J2NumPY.arange(y_min, y_max, h));
		double[][] YY = J2NumPY.meshgrid_getYY(J2NumPY.arange(x_min, x_max, h), J2NumPY.arange(y_min, y_max, h));

			
			 // Set class index
			 this.loadAttributesWithoutLabels(boundarySet, J2NumPY._c(J2NumPY.ravel(XX), J2NumPY.ravel(YY)));

		 try {
			cls.buildClassifier(trainingSet);
			
			Evaluation eval = new Evaluation(trainingSet);
			 eval.evaluateModel(cls, testingSet);
			 log.logln(G.LOG_FINE, eval.toSummaryString("\nResults\n======\n", false));
			 
			predLabels_Test = new double[testingSet.numInstances()];
			for (int index = 0; index < predLabels_Test.length; index++) {
				predLabels_Test[index] = cls.classifyInstance(testingSet.instance(index));
			}
		
			
			double[] predBound = new double[boundarySet.numInstances()];
			for (int index = 0; index < predBound.length; index++) {
				predBound[index] = cls.classifyInstance(boundarySet.instance(index));
			}
			predLabels_DescBound = J2NumPY.shape1D_2_2D(predBound, XX.length, XX[0].length);

			
			} catch (Exception e) {
				e.printStackTrace();
			}

	}

	public void loadTrainingAndPredict_CV_Bound(NDArray features_train, NDArray prediction, 
			double[] labels_train, double[] predict_labels) {
		 
		 // Create an empty training set
		 trainingSet = new Instances("terrain", alWekaAttributes, 10);
		 // Set class index
		 trainingSet.setClassIndex(2);
		 
		 System.out.println("Load training data.");
		 this.loadAttributes(trainingSet, features_train,  labels_train);

		 try {
			cls.buildClassifier(trainingSet);

		
	
		  // randomize data
		    Random rand = new Random(seed);
		    Instances randData = new Instances(trainingSet);
		    randData.randomize(rand);
		    if (randData.classAttribute().isNominal())
		      randData.stratify(folds);

		    // perform cross-validation and add predictions
		    Instances predictedData = null;
		    Evaluation eval = new Evaluation(randData);
		    for (int n = 0; n < folds; n++) {
		      Instances train = randData.trainCV(folds, n);
		      Instances test = randData.testCV(folds, n);
		      // the above code is used by the StratifiedRemoveFolds filter, the
		      // code below by the Explorer/Experimenter:
		      // Instances train = randData.trainCV(folds, n, rand);

		      // build and evaluate classifier
		      Classifier clsCopy = AbstractClassifier.makeCopy(cls);
		      clsCopy.buildClassifier(train);
		      eval.evaluateModel(clsCopy, test);

		      // add predictions
		      //AddClassification filter = new AddClassification();
		      filter.setClassifier(cls);
		      filter.setOutputClassification(true);
		      filter.setOutputDistribution(true);
		      filter.setOutputErrorFlag(true);
		      filter.setInputFormat(train);
		      Filter.useFilter(train, filter);  // trains the classifier
		      
		      
		      //fc.setFilter(rm);
		      fc.setClassifier(cls);
		      // train and make predictions
		      fc.buildClassifier(train);
		      Instances pred = Filter.useFilter(test, filter);  // perform predictions on test set
		      if (predictedData == null)
		        predictedData = new Instances(pred, 0);
		      for (int j = 0; j < pred.numInstances(); j++)
		        predictedData.add(pred.instance(j));
		    }
		    

		    // output evaluation
		    System.out.println();
		    System.out.println("=== Setup ===");
		    if (cls instanceof OptionHandler)
		      System.out.println("Classifier: " + cls.getClass().getName() + " " + Utils.joinOptions(((OptionHandler) cls).getOptions()));
		    else
		      System.out.println("Classifier: " + cls.getClass().getName());
		    System.out.println("Dataset: " + trainingSet.relationName());
		    System.out.println("Folds: " + folds);
		    System.out.println("Seed: " + seed);
		    System.out.println();
		    System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));

		//    Enumeration<Instance> enumPred = predictedData.enumerateInstances();
		//    while (enumPred.hasMoreElements()) {
		//    	Instance predVal = enumPred.nextElement();
		//    	predVal.
		//    }
		    
		    Iterator<Instance> predInstances = predictedData.iterator();
		    int index=0;
		    while (predInstances.hasNext()) {
		    	Instance predVal = predInstances.next();
		    	//Enumeration attributes = predVal.enumerateAttributes();
		    	System.out.print(predVal.value(0));
		    	System.out.print("\t:"+predVal.value(1));
		    	System.out.println("\t:"+predVal.value(2));
		    	prediction.set(index, 0, predVal.value(0));
		    	prediction.set(index, 1, predVal.value(1));
		    	predict_labels[index]=predVal.value(2);
		    	index++;
		    	//while (attributes.hasMoreElements()) {
		    	//	System.out.print("val: " + attributes.nextElement().toString());
		    	//	Attribute attr = (Attribute) attributes.nextElement();
		    	//	System.out.println("\t:" + attr.index() +"\t:" + attr.name() +"\t:"+attr.value(0));
		    	//}
		    
		    }
//			 Instance iTestExample = new DenseInstance(4);
//			 iTestExample.setValue((Attribute)fvWekaAttributes.get(0), 1.0);
		    // output "enriched" dataset
		    //DataSink.write(Utils.getOption("o", args), predictedData);
		    //create blank data
			double[][] XX = J2NumPY.meshgrid_getXX(J2NumPY.arange(x_min, x_max, h), J2NumPY.arange(y_min, y_max, h));
			double[][] YY = J2NumPY.meshgrid_getYY(J2NumPY.arange(x_min, x_max, h), J2NumPY.arange(y_min, y_max, h));
				
			Instances boundary = new Instances("boundary", alWekaAttributes, 10);
			 // Set class index
			 trainingSet.setClassIndex(2);
			 this.loadAttributesWithoutLabels(boundary, J2NumPY._c(J2NumPY.ravel(XX), J2NumPY.ravel(YY)));
			 Instances predBound = Filter.useFilter(boundary, filter);
			 Instances predictedBoundData =  new Instances(boundary, 0);
			      for (int j = 0; j < predBound.numInstances(); j++)
			        predictedBoundData.add(predBound.instance(j));

			double[] boundaryDescision = new double[predBound.numInstances()];
		    Iterator<Instance> predBoundInstances = predictedBoundData.iterator();
		    index=0;
		    while (predBoundInstances.hasNext()) {
		    	Instance predBoundVal = predBoundInstances.next();
		    	
		    	//Enumeration attributes = predVal.enumerateAttributes();
		    	//System.out.print(predBoundVal.value(0));
		    	//System.out.print("\t:"+predBoundVal.value(1));
		    	//System.out.println("\t:"+predBoundVal.value(2));
		    	System.out.print(predBoundVal.value(0));
		    	System.out.print("\t:"+predBoundVal.value(1));
		    	System.out.println("\t:"+predBoundVal.value(2));

		    	boundaryDescision[index]=predBoundVal.value(2);
		    	index++;
		    }
		    System.out.println("Boundary: " + ArrayUtils.printArray(boundaryDescision));
			//double[] boundaryDescision = gnb.predictDecisionBoundryData(J2NumPY._c(J2NumPY.ravel(XX), J2NumPY.ravel(YY)),labels_train);
			predLabels_DescBound = J2NumPY.shape1D_2_2D(boundaryDescision, XX.length, XX[0].length);

			} catch (Exception e) {
				e.printStackTrace();
			}

	}

	public double[][] getPredictedBoundaryDescision() {
		return this.predLabels_DescBound;
	}
	
	public double[] getPredictedTestingLabels() {
		return this.predLabels_Test;
	}
	
	public void loadTrainingAndPredict_Works(NDArray features_train, NDArray prediction, 
			double[] labels_train, double[] predict_labels) {
		 
		 // Create an empty training set
		 trainingSet = new Instances("terrain", alWekaAttributes, 10);
		 // Set class index
		 trainingSet.setClassIndex(2);
		 
		 System.out.println("Load training data.");
		 this.loadAttributes(trainingSet, features_train,  labels_train);

		 try {
			cls.buildClassifier(trainingSet);

		
	
		  // randomize data
		    Random rand = new Random(seed);
		    Instances randData = new Instances(trainingSet);
		    randData.randomize(rand);
		    if (randData.classAttribute().isNominal())
		      randData.stratify(folds);

		    // perform cross-validation and add predictions
		    Instances predictedData = null;
		    Evaluation eval = new Evaluation(randData);
		    for (int n = 0; n < folds; n++) {
		      Instances train = randData.trainCV(folds, n);
		      Instances test = randData.testCV(folds, n);
		      // the above code is used by the StratifiedRemoveFolds filter, the
		      // code below by the Explorer/Experimenter:
		      // Instances train = randData.trainCV(folds, n, rand);

		      // build and evaluate classifier
		      Classifier clsCopy = AbstractClassifier.makeCopy(cls);
		      clsCopy.buildClassifier(train);
		      eval.evaluateModel(clsCopy, test);

		      // add predictions
		      //AddClassification filter = new AddClassification();
		      filter.setClassifier(cls);
		      filter.setOutputClassification(true);
		      filter.setOutputDistribution(true);
		      filter.setOutputErrorFlag(true);
		      filter.setInputFormat(train);
		      Filter.useFilter(train, filter);  // trains the classifier
		      
		      
		      //fc.setFilter(rm);
		      fc.setClassifier(cls);
		      // train and make predictions
		      fc.buildClassifier(train);
		      Instances pred = Filter.useFilter(test, filter);  // perform predictions on test set
		      if (predictedData == null)
		        predictedData = new Instances(pred, 0);
		      for (int j = 0; j < pred.numInstances(); j++)
		        predictedData.add(pred.instance(j));
		    }

		    // output evaluation
		    System.out.println();
		    System.out.println("=== Setup ===");
		    if (cls instanceof OptionHandler)
		      System.out.println("Classifier: " + cls.getClass().getName() + " " + Utils.joinOptions(((OptionHandler) cls).getOptions()));
		    else
		      System.out.println("Classifier: " + cls.getClass().getName());
		    System.out.println("Dataset: " + trainingSet.relationName());
		    System.out.println("Folds: " + folds);
		    System.out.println("Seed: " + seed);
		    System.out.println();
		    System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));

		//    Enumeration<Instance> enumPred = predictedData.enumerateInstances();
		//    while (enumPred.hasMoreElements()) {
		//    	Instance predVal = enumPred.nextElement();
		//    	predVal.
		//    }
		    
		    Iterator<Instance> predInstances = predictedData.iterator();
		    int index=0;
		    while (predInstances.hasNext()) {
		    	Instance predVal = predInstances.next();
		    	//Enumeration attributes = predVal.enumerateAttributes();
		    	System.out.print(predVal.value(0));
		    	System.out.print("\t:"+predVal.value(1));
		    	System.out.println("\t:"+predVal.value(2));
		    	prediction.set(index, 0, predVal.value(0));
		    	prediction.set(index, 1, predVal.value(1));
		    	predict_labels[index]=predVal.value(2);
		    	index++;
		    	//while (attributes.hasMoreElements()) {
		    	//	System.out.print("val: " + attributes.nextElement().toString());
		    	//	Attribute attr = (Attribute) attributes.nextElement();
		    	//	System.out.println("\t:" + attr.index() +"\t:" + attr.name() +"\t:"+attr.value(0));
		    	//}
		    
		    }
//			 Instance iTestExample = new DenseInstance(4);
//			 iTestExample.setValue((Attribute)fvWekaAttributes.get(0), 1.0);
		    // output "enriched" dataset
		    //DataSink.write(Utils.getOption("o", args), predictedData);

			} catch (Exception e) {
				e.printStackTrace();
			}

	}

	public void loadTestingData(NDArray features_test, double[] labels_test) {
		 // Create an empty training set
		 Instances isTestingSet = new Instances("Frontier", alWekaAttributes, 10);
		 // Set class index
		 isTestingSet.setClassIndex(2);

		 System.out.println("\nLoad testing data.");
		 this.loadAttributes(isTestingSet, features_test,  labels_test);

		 // Test the model
		 Evaluation eTest;
		try {
			eTest = new Evaluation(trainingSet);
			eTest.evaluateModel(cls,  isTestingSet);
		 
		// Print the result à la Weka explorer:
		 String strSummary = eTest.toSummaryString();
		 System.out.println(strSummary);
		
		 // Get the confusion matrix
		 double[][] cmMatrix = eTest.confusionMatrix();
		 System.out.println("Confusion matrix:" + ArrayUtils.printArray(cmMatrix));
		
/*		NaiveBayes naiveBayes = new NaiveBayes();
		try {
			naiveBayes.buildClassifier(isTrainingSet);

        // this does the trick  
        double label = naiveBayes.classifyInstance(isTestingSet.instance(0));
        isTestingSet.instance(0).setClassValue(label);
*/
		 double label = cls.classifyInstance(isTestingSet.instance(0));
	        isTestingSet.instance(0).setClassValue(label);

        System.out.println("testing set predict size: " + isTestingSet.numInstances());
        Enumeration<Instance> eloop =  isTestingSet.enumerateInstances();
        while (eloop.hasMoreElements() ) {
        	Instance val = eloop.nextElement();
        	System.out.println(val.stringValue(2));
        }



//        System.out.println(isTestingSet.instance(0).stringValue(2));
//        System.out.println(isTestingSet.instance(1).stringValue(2));
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		 
	}
	
	public double[] predictDecisionBoundryData(NDArray features_test, double[] labels_test) 
	throws Exception {

		 // Create an empty training set
		 //Instances trainSet = new Instances("terrain", alWekaAttributes, 10);
		 // Set class index
		 //trainSet.setClassIndex(2);
		Instances descisionSet = new Instances ("boundary", alWekaAttributes,10);
		descisionSet.setClassIndex(2);
		 
		 System.out.println("Load terrain training data.");
		 this.loadAttributes(descisionSet, features_test,  labels_test);

		 //assuming trainingSet has already been created
		 try {
			 Classifier bnd = (Classifier)new NaiveBayes();
			 bnd.buildClassifier(trainingSet);
	
			} catch (Exception e) {
				e.printStackTrace();
			}

		  // randomize data
		    Random rand = new Random(seed);
		    Instances randData = new Instances(descisionSet);
		    randData.randomize(rand);
		    //if (randData.classAttribute().isNominal())
		    //  randData.stratify(folds);

		    // perform cross-validation and add predictions
		    Instances predictedData = null;
		    Evaluation eval = new Evaluation(randData);
//		    for (int n = 0; n < folds; n++) {
		      //Instances train = randData.trainCV(folds, n);
		    
		    //Instances bound = randData.testCV(2, 0);
		    Instances bound = descisionSet.testCV(2, 0);
		      //Instances bound = randData.testCV(folds, n);
		      // the above code is used by the StratifiedRemoveFolds filter, the
		      // code below by the Explorer/Experimenter:
		      // Instances train = randData.trainCV(folds, n, rand);

		      // build and evaluate classifier
		      //Classifier clsCopy = AbstractClassifier.makeCopy(cls);
		      //clsCopy.buildClassifier(train);
		      eval.evaluateModel(cls, bound);

		      // add predictions
/*		      AddClassification filter = new AddClassification();
		      filter.setClassifier(cls);
		      filter.setOutputClassification(true);
		      filter.setOutputDistribution(true);
		      filter.setOutputErrorFlag(true);
		      filter.setInputFormat(train);
		      Filter.useFilter(train, filter);  // trains the classifier
*/
		      Instances pred = Filter.useFilter(bound, filter);  // perform predictions on test set
		      if (predictedData == null)
		        predictedData = new Instances(pred, 0);
		      for (int j = 0; j < pred.numInstances(); j++)
		        predictedData.add(pred.instance(j));
		    
		      //fc.buildClassifier(train);
		      for (int i = 0; i < bound.numInstances(); i++) {
		        double predB = fc.classifyInstance(bound.instance(i));
		        System.out.print("ID: " + bound.instance(i).value(0));
		        System.out.print(", actual: " + bound.classAttribute().value((int) bound.instance(i).classValue()));
		        System.out.println(", predicted: " + bound.classAttribute().value((int) predB));
		      }
		    // output evaluation
		    System.out.println();
		    System.out.println("=== Setup ===");
		    if (cls instanceof OptionHandler)
		      System.out.println("Classifier: " + cls.getClass().getName() + " " + Utils.joinOptions(((OptionHandler) cls).getOptions()));
		    else
		      System.out.println("Classifier: " + cls.getClass().getName());
		    System.out.println("Dataset: " + descisionSet.relationName());
//		    System.out.println("Folds: " + folds);
//		    System.out.println("Seed: " + seed);
//		    System.out.println();
//		    System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));

		//    Enumeration<Instance> enumPred = predictedData.enumerateInstances();
		//    while (enumPred.hasMoreElements()) {
		//    	Instance predVal = enumPred.nextElement();
		//    	predVal.
		//    }

			double[] predict_labels = new double[descisionSet.numInstances()];

		    Iterator<Instance> predInstances = predictedData.iterator();
		    int index=0;
		    while (predInstances.hasNext()) {
		    	Instance predVal = predInstances.next();
		    	//Enumeration attributes = predVal.enumerateAttributes();
		    	System.out.print(predVal.value(0));
		    	System.out.print("\t:"+predVal.value(1));
		    	System.out.println("\t:"+predVal.value(2));
		    	predict_labels[index]=predVal.value(2);
		    	index++;
		    	//while (attributes.hasMoreElements()) {
		    	//	System.out.print("val: " + attributes.nextElement().toString());
		    	//	Attribute attr = (Attribute) attributes.nextElement();
		    	//	System.out.println("\t:" + attr.index() +"\t:" + attr.name() +"\t:"+attr.value(0));
		    	}
		    
	//	    }
//			 Instance iTestExample = new DenseInstance(4);
//			 iTestExample.setValue((Attribute)fvWekaAttributes.get(0), 1.0);
		    // output "enriched" dataset
		    //DataSink.write(Utils.getOption("o", args), predictedData);

	
		return predict_labels;

	}
	public double[] predictDecisionBoundryData_OLD1(NDArray features_train, double[] labels_train, NDArray features_test) 
	throws Exception {

		Instances descisionSet, trainSet;
		double[] predict_labels;
		// Create an empty training set
		descisionSet = new Instances("boundary", alWekaAttributes, 10);
		trainSet = new Instances("training", alWekaAttributes, 10);
		// Set class index
		descisionSet.setClassIndex(2);
		trainSet.setClassIndex(2);

		System.out.println("Load boundary data.");
		this.loadAttributes(trainSet, features_train, labels_train);
		this.loadAttributesWithoutLabels(descisionSet, features_test);

		Instances predictedData = null;
		Evaluation eval = new Evaluation(descisionSet);

		// build and evaluate classifier
		Classifier clsCopy = AbstractClassifier.makeCopy(cls);
		clsCopy.buildClassifier(trainSet);
		eval.evaluateModel(clsCopy, descisionSet);

		// add predictions
		AddClassification filter = new AddClassification();
		filter.setClassifier(cls);
		filter.setOutputClassification(true);
		filter.setOutputDistribution(true);
		filter.setOutputErrorFlag(true);
		filter.setInputFormat(descisionSet);
		Filter.useFilter(trainSet, filter); // trains the classifier
		Instances pred = Filter.useFilter(descisionSet, filter); // perform
																	// predictions
																	// on test
																	// set
		if (predictedData == null)
			predictedData = new Instances(pred, 0);
		for (int j = 0; j < pred.numInstances(); j++)
			predictedData.add(pred.instance(j));

		predict_labels = new double[descisionSet.numInstances()];
		Iterator<Instance> predInstances = predictedData.iterator();
		int index = 0;
		while (predInstances.hasNext()) {
			Instance predVal = predInstances.next();
			// Enumeration attributes = predVal.enumerateAttributes();
			System.out.print(predVal.value(0));
			System.out.print("\t:" + predVal.value(1));
			System.out.println("\t:" + predVal.value(2));
			// prediction.set(index, 0, predVal.value(0));
			// prediction.set(index, 1, predVal.value(1));
			predict_labels[index] = predVal.value(2);
			index++;
			// while (attributes.hasMoreElements()) {
			// System.out.print("val: " + attributes.nextElement().toString());
			// Attribute attr = (Attribute) attributes.nextElement();
			// System.out.println("\t:" + attr.index() +"\t:" + attr.name()
			// +"\t:"+attr.value(0));
			// }

		}
		// Instance iTestExample = new DenseInstance(4);
		// iTestExample.setValue((Attribute)fvWekaAttributes.get(0), 1.0);
		// output "enriched" dataset
		// DataSink.write(Utils.getOption("o", args), predictedData);

		return predict_labels;

	}

	public double[] predictDecisionBoundryData_OLD2(NDArray features_test) {
		//same as load data, not if sure yet if all instances remain so create a copy
		//and predict against that one
		
		
		 // Create an empty training set
		 Instances isDescSet = new Instances("Rel", alWekaAttributes, 10);
		 // Set class index
		 isDescSet.setClassIndex(2);

		 System.out.println("\nLoad desc data.");
		 this.loadAttributesWithoutLabels(isDescSet, features_test);

	 // Test the model
		 Evaluation eTest;
		try {
			eTest = new Evaluation(trainingSet);
			eTest.evaluateModel(cls,  isDescSet);
		 
		// Print the result à la Weka explorer:
		 String strSummary = eTest.toSummaryString();
		 System.out.println(strSummary);
		
		 // Get the confusion matrix
		 double[][] cmMatrix = eTest.confusionMatrix();
		 System.out.println("Confusion matrix:" + ArrayUtils.printArray(cmMatrix));
		
		 Classifier descModel = (Classifier)new NaiveBayes();
		descModel.buildClassifier(trainingSet);
 
		 double label = descModel.classifyInstance(isDescSet.instance(0));
	        isDescSet.instance(0).setClassValue(label);
		} catch (Exception e) {
			e.printStackTrace();
		}
       System.out.println("descision set predict size: " + isDescSet.numInstances());
       Enumeration<Instance> eloop =  isDescSet.enumerateInstances();
		double[] predict = new double[isDescSet.numInstances()];
       int index=0;
       while (eloop.hasMoreElements() ) {
       	Instance val = eloop.nextElement();
       	//System.out.println(val.stringValue(2));
       	predict[index] = Integer.parseInt(val.stringValue(2));
       	index++;
       }
       

       
//       System.out.println(isTestingSet.instance(0).stringValue(2));
//       System.out.println(isTestingSet.instance(1).stringValue(2));

return predict;		
		 
	}
	
	private void loadAttributes(Instances instancesSet, NDArray features, double[] labels) {
		// Create the instance
		Iterator<INDArray> iLoop = features.iterator();
		int lblCount = 0;
		
		while (iLoop.hasNext() && lblCount < labels.length ) {
			INDArray value = iLoop.next();
			double x = value.get(0);
			double y = value.get(1);
			
		 Instance iExample = new DenseInstance(3);
		 iExample.setValue((Attribute)alWekaAttributes.get(0), x);
		 iExample.setValue((Attribute)alWekaAttributes.get(1), y);
		 
		 //System.out.print("x: " + x +"\ty: " + y +"\tclass: ");
		 if (labels[lblCount] == 0) {
			 iExample.setValue((Attribute)alWekaAttributes.get(2), WekaClassifiers.sFast);
			 //System.out.println("fast");
		 } else {
			 iExample.setValue((Attribute)alWekaAttributes.get(2), WekaClassifiers.sSlow);
			 //System.out.println("slow");
		 } 
		 
		
		 // add the instance
		 instancesSet.add(iExample);
		 lblCount++;
		}
		
	}
	
	private void loadAttributesWithoutLabels(Instances instancesSet, NDArray features) {
		// Create the instance
		Iterator<INDArray> iLoop = features.iterator();
		int lblCount = 0;
		
		while (iLoop.hasNext() ) {
			INDArray value = iLoop.next();
			double x = value.get(0);
			double y = value.get(1);
			
		 Instance iExample = new DenseInstance(3);
		 iExample.setValue((Attribute)alWekaAttributes.get(0), x);
		 iExample.setValue((Attribute)alWekaAttributes.get(1), y);
		 //does it matters what labels are when it is testing data?
		 iExample.setValue((Attribute)alWekaAttributes.get(2), WekaClassifiers.sFast);
		 	
		 // add the instance
		 instancesSet.add(iExample);
		 lblCount++;
		}
		
	}

	public double[] getPredictionLabels(Instances testingSet) {
		double[] pred = new double[testingSet.numInstances()];
		
		Enumeration<Instance> eloop =  testingSet.enumerateInstances();
		int index=0;
        while (eloop.hasMoreElements() ) {
        	Instance val = eloop.nextElement();
        	pred[index] = Integer.parseInt(val.stringValue(2));
        	//System.out.println(val.stringValue(2));
        	index++;
        }
        		
		return pred;
	}

	public void loadData_Working() {
		// Declare two numeric attributes
		 Attribute Attribute1 = new Attribute("firstNumeric");
		 Attribute Attribute2 = new Attribute("secondNumeric");

		 // Declare a nominal attribute along with its values
		 //FastVector fvNominalVal = new FastVector(3);
		 ArrayList<String> fvNominalVal = new ArrayList<String>();
		 fvNominalVal.add("blue");
		 fvNominalVal.add("gray");
		 fvNominalVal.add("black");
		 
		 Attribute Attribute3 = new Attribute("aNominal", fvNominalVal); 
		
		// Declare the class attribute along with its values
	//	 FastVector fvClassVal = new FastVector(2);
		 ArrayList<String> fvClassVal = new ArrayList<String>();
		 fvClassVal.add("positive");
		 fvClassVal.add("negative");

		 Attribute ClassAttribute = new Attribute("theClass", fvClassVal); 
		
		 // Declare the feature vector
	//	 FastVector<Attribute> fvWekaAttributes = new FastVector(4);
		 ArrayList<Attribute> fvWekaAttributes = new ArrayList<Attribute>();
		 fvWekaAttributes.add(Attribute1);
		 fvWekaAttributes.add(Attribute2);
		 fvWekaAttributes.add(Attribute3);
		 fvWekaAttributes.add(ClassAttribute);
		 
		 // Create an empty training set
		 Instances isTrainingSet = new Instances("Rel", fvWekaAttributes, 10);
		 // Set class index
		 isTrainingSet.setClassIndex(3);
		 
		// Create the instance
		 Instance iExample = new DenseInstance(4);
		 iExample.setValue((Attribute)fvWekaAttributes.get(0), 1.0);
		 iExample.setValue((Attribute)fvWekaAttributes.get(1), 0.5);
		 iExample.setValue((Attribute)fvWekaAttributes.get(2), "gray");
		 iExample.setValue((Attribute)fvWekaAttributes.get(3), "positive");
		
		 // add the instance
		 isTrainingSet.add(iExample);
		 		 
		 // Create a naïve bayes classifier
		 Classifier cModel = (Classifier)new NaiveBayes();
		 try {
			cModel.buildClassifier(isTrainingSet);
		} catch (Exception e) {
			e.printStackTrace();
		}

		 // Create an empty training set
		 Instances isTestingSet = new Instances("Rel", fvWekaAttributes, 10);
		 // Set class index
		 isTestingSet.setClassIndex(3);
		 
			// Create the instance
		 Instance iTestExample = new DenseInstance(4);
		 iTestExample.setValue((Attribute)fvWekaAttributes.get(0), 1.0);
		 iTestExample.setValue((Attribute)fvWekaAttributes.get(1), 0.5);
		 iTestExample.setValue((Attribute)fvWekaAttributes.get(2), "gray");
		 iTestExample.setValue((Attribute)fvWekaAttributes.get(3), "negative");

		 isTestingSet.add(iTestExample);

		 // Test the model
		 Evaluation eTest;
		try {
			eTest = new Evaluation(isTrainingSet);
			eTest.evaluateModel(cModel,  isTestingSet);
		 
		// Print the result à la Weka explorer:
		 String strSummary = eTest.toSummaryString();
		 System.out.println(strSummary);
		
		 // Get the confusion matrix
		 double[][] cmMatrix = eTest.confusionMatrix();
		} catch (Exception e) {
			e.printStackTrace();
		}
		 
	}

	public void trySVM(NDArray features_train, NDArray features_test, 
			double[] labels_train, double[] dLabels_test) {
		try {
	//	WekaPackageManager.loadPackages( false, true, false );
		AbstractClassifier classifier = ( AbstractClassifier ) Class.forName(
		            "weka.classifiers.functions.LibSVM" ).newInstance();
	/*
	 Valid options are: 


 -S <int>
  Set type of SVM (default: 0)
    0 = C-SVC
    1 = nu-SVC
    2 = one-class SVM
    3 = epsilon-SVR
    4 = nu-SVR
 -K <int>
  Set type of kernel function (default: 2)
    0 = linear: u'*v
    1 = polynomial: (gamma*u'*v + coef0)^degree
    2 = radial basis function: exp(-gamma*|u-v|^2)
    3 = sigmoid: tanh(gamma*u'*v + coef0)
 -D <int>
  Set degree in kernel function (default: 3)
 -G <double>
  Set gamma in kernel function (default: 1/k)
 -R <double>
  Set coef0 in kernel function (default: 0)
 -C <double>
  Set the parameter C of C-SVC, epsilon-SVR, and nu-SVR
   (default: 1)
 -N <double>
  Set the parameter nu of nu-SVC, one-class SVM, and nu-SVR
   (default: 0.5)
 -Z
  Turns on normalization of input data (default: off)
 -J
  Turn off nominal to binary conversion.
  WARNING: use only if your data is all numeric!
 -V
  Turn off missing value replacement.
  WARNING: use only if your data has no missing values.
 -P <double>
  Set the epsilon in loss function of epsilon-SVR (default: 0.1)
 -M <double>
  Set cache memory size in MB (default: 40)
 -E <double>
  Set tolerance of termination criterion (default: 0.001)
 -H
  Turns the shrinking heuristics off (default: on)
 -W <double>
  Set the parameters C of class i to weight[i]*C, for C-SVC.
  E.g., for a 3-class problem, you could use "1 1 1" for equally
  weighted classes.
  (default: 1 for all classes)
 -B
  Trains a SVC model instead of a SVR one (default: SVR)
 -model <file>
  Specifies the filename to save the libsvm-internal model to.
  Gets ignored if a directory is provided.
 -D
  If set, classifier is run in debug mode and
  may output additional info to the console
 -seed <num>
  Seed for the random number generator when -B is used.
  (default = 1)
	
	 */
//		String options = ( "-S 0 -K 2 -D 3 -G 1000.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1" );
		String options = ( "-S 0 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1000.0 -E 0.001 -P 0.1" );
		String[] optionsArray = options.split( " " );
		    
		    
	
	
	 // Create an empty training set
	 trainingSet = new Instances("terrain", alWekaAttributes, 10);
	 testingSet = new Instances("testing", alWekaAttributes, 10);
	 boundarySet = new Instances("boundary", alWekaAttributes, 10);
	 // Set class index
	 trainingSet.setClassIndex(2);
	 testingSet.setClassIndex(2);
	 boundarySet.setClassIndex(2);
	 
	 this.loadAttributes(trainingSet, features_train,  labels_train);
	 this.loadAttributes(testingSet, features_test,  dLabels_test);

	 double[][] XX = J2NumPY.meshgrid_getXX(J2NumPY.arange(x_min, x_max, h), J2NumPY.arange(y_min, y_max, h));
	double[][] YY = J2NumPY.meshgrid_getYY(J2NumPY.arange(x_min, x_max, h), J2NumPY.arange(y_min, y_max, h));

		
		 // Set class index
		 this.loadAttributesWithoutLabels(boundarySet, J2NumPY._c(J2NumPY.ravel(XX), J2NumPY.ravel(YY)));

	 
		//cls.buildClassifier(trainingSet);
		 classifier.setOptions( optionsArray );
		 classifier.buildClassifier( trainingSet );	
		Evaluation eval = new Evaluation(trainingSet);
		 eval.evaluateModel(classifier, testingSet);
		 log.logln(G.LOG_FINE, eval.toSummaryString("\nResults\n======\n", false));
		 
	/*	predLabels_Test = new double[testingSet.numInstances()];
		for (int index = 0; index < predLabels_Test.length; index++) {
			predLabels_Test[index] = classifier.classifyInstance(testingSet.instance(index));
		}
	*/
		 predLabels_Test = WekaUtils.getPredictedLabels(testingSet, classifier);
		
	/*	double[] predBound = new double[boundarySet.numInstances()];
		for (int index = 0; index < predBound.length; index++) {
			predBound[index] = classifier.classifyInstance(boundarySet.instance(index));
		}
		predLabels_DescBound = J2NumPY.shape1D_2_2D(predBound, XX.length, XX[0].length);
*/
		 double[] predBound = WekaUtils.getPredictedLabels(boundarySet, classifier);
			predLabels_DescBound = J2NumPY.shape1D_2_2D(predBound, XX.length, XX[0].length);

			
		} catch (Exception e) {
			e.printStackTrace();
		}


	}

}
