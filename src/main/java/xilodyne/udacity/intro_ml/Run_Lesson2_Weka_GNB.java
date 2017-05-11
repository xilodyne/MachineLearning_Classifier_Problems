package xilodyne.udacity.intro_ml;

import xilodyne.udacity.intro_ml.lesson2_svm_weka_terraindata.StudentMain;
import xilodyne.util.G;
import xilodyne.util.Logger;

public class Run_Lesson2_Weka_GNB {
	private static Logger log = new Logger();

	public static void main(String[] args) {

		 G.setLoggerLevel(G.LOG_FINE);
		// G.setLoggerLevel(G.LOG_INFO);
		//G.setLoggerLevel(G.LOG_DEBUG);
		// G.setLoggerShowDate(false);

		log.logln_withClassName(G.lF, "Running Gaussian Naive Bayes classification.");
		StudentMain studentMainND = new StudentMain();
		//studentMainND.doNB();
		studentMainND.doNB_Weka();
		//studentMainND.doNB_generateARFF();
		//studentMainND.doWeka_GNB();
	}
}
