package xilodyne.udacity.intro_ml;

import xilodyne.udacity.intro_ml.lesson2_svm_weka_terraindata.StudentMain;
import xilodyne.util.G;
import xilodyne.util.Logger;

/**
 * Java implementation of Gaussian NB terrain data example from Udacity. From
 * Udacity course (in python): Intro to Machine Learning.
 * 
 * @see <a
 *      href="https://www.udacity.com/course/intro-to-machine-learning--ud120">https://www.udacity.com/course/intro-to-machine-learning--ud120</a>
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.1
 * 
 */
public class Run_Lesson2_Perceptron {

	private static Logger log = new Logger();

	public static void main(String[] args) {

		 G.setLoggerLevel(G.LOG_FINE);
		// G.setLoggerLevel(G.LOG_INFO);
		//G.setLoggerLevel(G.LOG_DEBUG);
		// G.setLoggerShowDate(false);

		log.logln_withClassName(G.lF, "Running Gaussian Naive Bayes classification.");
		StudentMain studentMainND = new StudentMain();
		//studentMainND.doNB();
		//studentMainND.doNB_Weka();
		//studentMainND.doNB_generateARFF();
		//studentMainND.doNB_skipARFF();
		studentMainND.doPerceptron();
	}

}
