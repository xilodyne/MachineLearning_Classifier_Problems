package xilodyne.udacity.intro_ml.lesson2_svm_weka_terraindata;

import mikera.arrayz.NDArray;
import xilodyne.util.ArrayUtils;
import xilodyne.util.G;
import xilodyne.util.Logger;

/**
 * Java implementation of the python prep_terrain_data.py from the Udacity Intro
 * to Machine Learning course.
 * 
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.1
 */

public class Prep_terrain_data {

	private Logger log = new Logger();

	/*
	 * make the toy dataset
	 */
	public void makeTerrainData(NDArray X_train, NDArray X_test, double[] y_train, double[] y_test) {

		double grade[] = new double[StudentMain.size];
		double bumpy[] = new double[StudentMain.size];
		double error[] = new double[StudentMain.size];
		double y[] = new double[StudentMain.size];

		// generate dummy data
		for (int x = 0; x < StudentMain.size; x++) {
			grade[x] = Math.random();
			bumpy[x] = Math.random();
			error[x] = Math.random();
		}

		// combine y dummy data
		for (int loop = 0; loop < StudentMain.size; loop++)
			y[loop] = (double) Math.round((grade[loop] * bumpy[loop]) + 0.3 + (0.1 * error[loop]));

		log.logln_withClassName(G.lF, "");
		log.logln("Created data.");
		log.logln(G.lD, "\nGrade: " + ArrayUtils.printArray(grade));
		log.logln("\nBumpy: " + ArrayUtils.printArray(bumpy));
		log.logln("\nError: " + ArrayUtils.printArray(error));
		log.logln("\ny: " + ArrayUtils.printArray(y));

		for (int loop = 0; loop < StudentMain.size; loop++)
			if (grade[loop] > 0.8 || bumpy[loop] > 0.8)
				y[loop] = 1.0;

		log.logln_noTimestamp("y: " + ArrayUtils.printArray(y));

		// create training subset
		for (int loop = 0; loop < StudentMain.split; loop++) {
			X_train.set(loop, 0, grade[loop]);
			X_train.set(loop, 1, bumpy[loop]);
			y_train[loop] = y[loop];
		}

		for (int loop = 0; loop < StudentMain.size - StudentMain.split; loop++) {
			X_test.set(loop, 0, grade[loop + StudentMain.split]);
			X_test.set(loop, 1, bumpy[loop + StudentMain.split]);
			y_test[loop] = y[loop + StudentMain.split];
		}

		// logger.log(Level.INFO, sClassName + ":X_train: " + X_train);
		// logger.log(Level.SEVERE, ex.toString(), ex);

		log.logln("\nX_train: " + X_train);
		log.logln("\nX_test:  " + X_test);
		log.logln("\ny_train: " + ArrayUtils.printArray(y_train));
		log.logln("\ny_test: " + ArrayUtils.printArray(y_test));

	}
}
