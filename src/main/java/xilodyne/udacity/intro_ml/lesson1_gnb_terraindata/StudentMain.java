package xilodyne.udacity.intro_ml.lesson1_gnb_terraindata;

import xilodyne.util.ArrayUtils;
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
	//public static final int size = 10;
	//public static final int size = 1000;
	public static final int size = 10000;
	private Logger log = new Logger();

	public void doNB() {

		log.logln_withClassName(G.lF, "");

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

		ClassifyNB clf = new ClassifyNB();
		log.logln_withClassName(G.lF, "Training data.");
		clf.classify(features_train, labels_train);
		double[] pred = clf.predict(features_test);
		log.logln(G.lI, "predict size: " + pred.length + ", test size: " + labels_test.length);
		log.logln(G.lD, "\npredict: " + ArrayUtils.printArray(pred));
		log.logln(G.lD, "\nlabels_train: " + ArrayUtils.printArray(labels_train));
		double accuracy = clf.getAccuracy(labels_test, pred);
		log.logln(G.lF, "Accuracy: " + accuracy);

		Class_vis cvis = new Class_vis();
		cvis.prettyPicture(clf, features_test, pred);

		cvis.output_image();

	}

}
