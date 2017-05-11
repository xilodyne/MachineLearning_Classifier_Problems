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

public class WekaClassifiers {
	
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
	
	public WekaClassifiers() {
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
	//	 cls = (Classifier)new NaiveBayes();
	//	 filter = new AddClassification();
	//	 fc = new FilteredClassifier();
	}
	
	public void runWeka_GNB(NDArray features_train, NDArray features_test, 
			double[] labels_train, double[] dLabels_test) {
		try {
	//	WekaPackageManager.loadPackages( false, true, false );
		AbstractClassifier classifier = ( AbstractClassifier ) Class.forName(
		            "weka.classifiers.bayes.BayesianLogisticRegression" ).newInstance();
		
		String options ="";
		
		this.loadData(classifier, options, features_train, features_test, labels_train, dLabels_test);

		} catch (Exception e) {
			e.printStackTrace();
		}
	}
	
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

	public void runWeka_SVM(NDArray features_train, NDArray features_test, 
			double[] labels_train, double[] dLabels_test)  {
		try {
	//	WekaPackageManager.loadPackages( false, true, false );
		AbstractClassifier classifier = ( AbstractClassifier ) Class.forName(
		            "weka.classifiers.functions.LibSVM" ).newInstance();
//		String options = ( "-S 0 -K 2 -D 3 -G 1000.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1" );
		String options = ( "-S 0 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1000.0 -E 0.001 -P 0.1" );
		
		this.loadData(classifier, options, features_train, features_test, labels_train, dLabels_test);
		
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



//       System.out.println(isTestingSet.instance(0).stringValue(2));
//       System.out.println(isTestingSet.instance(1).stringValue(2));
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		 
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
	
	private void loadData(AbstractClassifier classifier, String options, NDArray features_train, NDArray features_test, 
			double[] labels_train, double[] dLabels_test ) throws Exception {
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

	}
	
	
	public double[][] getPredictedBoundaryDescision() {
		return this.predLabels_DescBound;
	}
	
	public double[] getPredictedTestingLabels() {
		return this.predLabels_Test;
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
	

	


}
