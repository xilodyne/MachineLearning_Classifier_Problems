package xilodyne.udacity.intro_ml.lesson2_svm_weka_terraindata;

//import weka.core.Instances;
//import weka.core.Utils;
//import weka.core.converters.ConverterUtils.DataSource;
//import xilodyne.machinelearning.classifier.GaussianNB;
import xilodyne.machinelearning.classifier.neural.Perceptron;
import xilodyne.util.ArrayUtils;
import xilodyne.util.weka.WekaARFFUtils;
import xilodyne.util.G;
import xilodyne.util.Logger;
import mikera.arrayz.NDArray;




/**
 * Java implementation of the python studentMain.py from the Udacity Intro to
 * Machine Learning course.
 * 
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.2
 */

public class StudentMain {
	public static int split = 0;
//	public static final int size = 1000;
//	public static final int size = 10;
	public static final int size = 10000;
	private Logger log = new Logger();

	public void doNB_Weka() {
		log.logln_withClassName(G.LOG_FINE, "");

		// setup
		StudentMain.split = (int) (0.75 * StudentMain.size);

		Prep_terrain_data prep_data = new Prep_terrain_data();

		NDArray features_train = NDArray.newArray(split, 2);
		NDArray features_test = NDArray.newArray(StudentMain.size - StudentMain.split, 2);
		double[] labels_train = new double[StudentMain.split];
		double[] labels_test = new double[StudentMain.size - StudentMain.split];

		// get dummy data
		prep_data.makeTerrainData(features_train, features_test, labels_train, labels_test);

		log.logln_withClassName(G.LOG_INFO, "Split: " + StudentMain.split);
		log.logln("");
		log.logln(G.LOG_FINE, "Size:\tf_train\tl_train\tf_test\tl_test");
		log.log("\t\t" + features_train.getShape(0));
		log.log_noTimestamp("\t" + labels_train.length);
		log.log_noTimestamp("\t" + features_test.getShape(0));
		log.logln_noTimestamp("\t" + labels_test.length);
		log.logln(G.LOG_DEBUG, "\nfeatures_train: " + features_train);
		log.logln("\nfeatures_test:  " + features_test);
		log.logln("\nlabels_train: " + ArrayUtils.printArray(labels_train));
		log.logln("\nlabels_test: " + ArrayUtils.printArray(labels_test));

		WekaClassifiers clf = new WekaClassifiers();
		clf.loadTrainingData(features_train, labels_train);
		clf.loadTestingData(features_test, labels_test);
	
		Class_vis cvis = new Class_vis();
		//cvis.prettyPicture(clf, features_train, labels_train, features_test, labels_test);
		cvis.prettyPicture(clf, features_test, labels_test);

		cvis.output_image();
		
	}
	
	
	
	/*public void doNB_Weka() {
		WekaPackageManager.loadPackages( false, true, false );
		try {
			AbstractClassifier classifier = ( AbstractClassifier ) Class.forName(
			            "weka.classifiers." ).newInstance();
		} catch (InstantiationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
*/

	
	/*
	public void doNB_generateARFF() {

		log.logln_withClassName(G.LOG_FINE, "");

		// setup
		StudentMain.split = (int) (0.75 * StudentMain.size);

		Prep_terrain_data prep_data = new Prep_terrain_data();

		NDArray features_train = NDArray.newArray(split, 2);
		NDArray features_test = NDArray.newArray(StudentMain.size - StudentMain.split, 2);
		double[] labels_train = new double[StudentMain.split];
		double[] labels_test = new double[StudentMain.size - StudentMain.split];
		double[][] features_train_2d = new double[2][split];

		// get dummy data
		prep_data.makeTerrainData(features_train, features_test, labels_train, labels_test);

		log.logln_withClassName(G.LOG_INFO, "Split: " + StudentMain.split);
		log.logln("");
		log.logln(G.LOG_FINE, "Size:\tf_train\tl_train\tf_test\tl_test");
		log.log("\t\t" + features_train.getShape(0));
		log.log_noTimestamp("\t" + labels_train.length);
		log.log_noTimestamp("\t" + features_test.getShape(0));
		log.logln_noTimestamp("\t" + labels_test.length);
		log.logln(G.LOG_DEBUG, "\nfeatures_train: " + features_train);
		log.logln("\nfeatures_test:  " + features_test);
		log.logln("\nlabels_train: " + ArrayUtils.print1DArray(labels_train));
		log.logln("\nlabels_test: " + ArrayUtils.print1DArray(labels_test));

//		ClassifyNB clf = new ClassifyNB();
//		log.logln_withClassName(G.lF, "Training data.");
//		clf.classify(features_train, labels_train);
		GaussianNB clf = new GaussianNB(GaussianNB.EMPTY_SAMPLES_IGNORE);
		try {
			clf.fit(features_train, labels_train);
		} catch (Exception e) {
			e.printStackTrace();
		}
		double[] pred = clf.predict(features_test);
		log.logln(G.lI, "predict size: " + pred.length + ", test size: " + labels_test.length);
		log.logln(G.lD, "\npredict: " + ArrayUtils.printArray(pred));
		log.logln(G.lD, "\nlabels_train: " + ArrayUtils.printArray(labels_train));
		double accuracy = clf.getAccuracyOfPredictedResults(labels_test, pred);
		log.logln(G.lF, "Accuracy: " + accuracy);


		//WEKA_ARFF_Utils.writeNDArrayToARFF(features_train, labels_train, "TerrainData_Train.arff");
		NDArray weka_predict = NDArray.newArray(split, 2);
		double[] weka_labels = new double[labels_train.length];
		WEKA_ARFF_Utils.readARFFdata(weka_predict, weka_labels, "predictions.arff");

		System.out.println("weka_predict: "+ weka_predict);
		System.out.println("weka_labels: " + ArrayUtils.print1DArray(weka_labels));
		
		Class_vis cvis = new Class_vis();
		cvis.prettyPicture(clf, weka_predict, weka_labels);

		cvis.output_image();

	}
	*/
	
	public void doSVM() {

		log.logln_withClassName(G.LOG_FINE, "");

		// setup
		StudentMain.split = (int) (0.75 * StudentMain.size);

		Prep_terrain_data prep_data = new Prep_terrain_data();

		NDArray features_train = NDArray.newArray(split, 2);
		NDArray features_test = NDArray.newArray(StudentMain.size - StudentMain.split, 2);
		double[] labels_train = new double[StudentMain.split];
		double[] labels_test = new double[StudentMain.size - StudentMain.split];
//		double[][] features_train_2d = new double[2][split];

		// get dummy data
		prep_data.makeTerrainData(features_train, features_test, labels_train, labels_test);

		log.logln_withClassName(G.LOG_INFO, "Split: " + StudentMain.split);
		log.logln("");
		log.logln(G.LOG_FINE, "Size:\tf_train\tl_train\tf_test\tl_test");
		log.log("\t\t" + features_train.getShape(0));
		log.log_noTimestamp("\t" + labels_train.length);
		log.log_noTimestamp("\t" + features_test.getShape(0));
		log.logln_noTimestamp("\t" + labels_test.length);
		log.logln(G.LOG_DEBUG, "\nfeatures_train: " + features_train);
		log.logln("\nfeatures_test:  " + features_test);
		log.logln("\nlabels_train: " + ArrayUtils.printArray(labels_train));
		log.logln("\nlabels_test: " + ArrayUtils.printArray(labels_test));


		WekaClassifiers weka = new WekaClassifiers();
		
	//	NDArray weka_predict = NDArray.newArray(split, 2);
	//	double[] pred_labels = new double[labels_train.length];
		
//		weka.loadTrainingAndPredict(features_train, features_test, labels_train, labels_test);
		weka.runWeka_SVM(features_train, features_test, labels_train, labels_test);
//		System.out.println("weka_predict: "+ weka_predict);
//		System.out.println("pred_labels: " + ArrayUtils.print1DArray(pred_labels));
		
		Class_vis cvis = new Class_vis();
		cvis.prettyPicture(weka, features_test);

		cvis.output_image();

	}
	
	public void doWeka_GNB() {

		log.logln_withClassName(G.LOG_FINE, "");

		// setup
		StudentMain.split = (int) (0.75 * StudentMain.size);

		Prep_terrain_data prep_data = new Prep_terrain_data();

		NDArray features_train = NDArray.newArray(split, 2);
		NDArray features_test = NDArray.newArray(StudentMain.size - StudentMain.split, 2);
		double[] labels_train = new double[StudentMain.split];
		double[] labels_test = new double[StudentMain.size - StudentMain.split];
//		double[][] features_train_2d = new double[2][split];

		// get dummy data
		prep_data.makeTerrainData(features_train, features_test, labels_train, labels_test);

		log.logln_withClassName(G.LOG_INFO, "Split: " + StudentMain.split);
		log.logln("");
		log.logln(G.LOG_FINE, "Size:\tf_train\tl_train\tf_test\tl_test");
		log.log("\t\t" + features_train.getShape(0));
		log.log_noTimestamp("\t" + labels_train.length);
		log.log_noTimestamp("\t" + features_test.getShape(0));
		log.logln_noTimestamp("\t" + labels_test.length);
		log.logln(G.LOG_DEBUG, "\nfeatures_train: " + features_train);
		log.logln("\nfeatures_test:  " + features_test);
		log.logln("\nlabels_train: " + ArrayUtils.printArray(labels_train));
		log.logln("\nlabels_test: " + ArrayUtils.printArray(labels_test));


		WekaClassifiers weka = new WekaClassifiers();
		
	//	NDArray weka_predict = NDArray.newArray(split, 2);
	//	double[] pred_labels = new double[labels_train.length];
		
//		weka.loadTrainingAndPredict(features_train, features_test, labels_train, labels_test);
		weka.runWeka_GNB(features_train, features_test, labels_train, labels_test);
//		System.out.println("weka_predict: "+ weka_predict);
//		System.out.println("pred_labels: " + ArrayUtils.print1DArray(pred_labels));
		
		Class_vis cvis = new Class_vis();
		cvis.prettyPicture(weka, features_test);

		cvis.output_image();

	}
	
	public void doPerceptron() {

		log.logln_withClassName(G.LOG_FINE, "");

		// setup
		StudentMain.split = (int) (0.75 * StudentMain.size);

		Prep_terrain_data prep_data = new Prep_terrain_data();

		NDArray features_train = NDArray.newArray(split, 2);
		NDArray features_test = NDArray.newArray(StudentMain.size - StudentMain.split, 2);
		double[] labels_train = new double[StudentMain.split];
		double[] labels_test = new double[StudentMain.size - StudentMain.split];

		// get dummy data
		prep_data.makeTerrainData(features_train, features_test, labels_train, labels_test);

		log.logln_withClassName(G.LOG_INFO, "Split: " + StudentMain.split);
		log.logln("");
		log.logln(G.LOG_FINE, "Size:\tf_train\tl_train\tf_test\tl_test");
		log.log("\t\t" + features_train.getShape(0));
		log.log_noTimestamp("\t" + labels_train.length);
		log.log_noTimestamp("\t" + features_test.getShape(0));
		log.logln_noTimestamp("\t" + labels_test.length);
		log.logln(G.LOG_DEBUG, "\nfeatures_train: " + features_train);
		log.logln("\nfeatures_test:  " + features_test);
		log.logln("\nlabels_train: " + ArrayUtils.printArray(labels_train));
		log.logln("\nlabels_test: " + ArrayUtils.printArray(labels_test));

		//ClassifyNB clf = new ClassifyNB();
		//log.logln_withClassName(G.lF, "Training data.");
		//clf.classify(features_train, labels_train);
		float fThreshold = 0;
		float[] fWeights = new float[] { 0, 0 };

		Perceptron clf = new Perceptron(fWeights, fThreshold);
		try {
			clf.fit(features_train, labels_train);
		} catch (Exception e) {
			e.printStackTrace();
		}
		double[] pred = clf.predict(features_test);
		log.logln(G.lI, "predict size: " + pred.length + ", test size: " + labels_test.length);
		log.logln(G.lD, "\npredict: " + ArrayUtils.printArray(pred));
		log.logln(G.lD, "\nlabels_train: " + ArrayUtils.printArray(labels_train));
		double accuracy = clf.getAccuracyOfPredictedResults(labels_test, pred);
		log.logln(G.lF, "Accuracy: " + accuracy);

		Class_vis cvis = new Class_vis();
		//cvis.prettyPicture(clf, features_test, pred);

	//	cvis.output_image();

	}
	

}
