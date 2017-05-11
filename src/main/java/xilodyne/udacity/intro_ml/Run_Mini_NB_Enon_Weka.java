package xilodyne.udacity.intro_ml;

import xilodyne.udacity.intro_ml.miniproj_NB_Enron.naive_bayes.nb_author_id;
import xilodyne.udacity.intro_ml.miniproj_NB_Enron.weka.WekaEnron;
import xilodyne.util.G;
import xilodyne.util.Logger;

public class Run_Mini_NB_Enon_Weka {

	private static Logger log = new Logger();

	public static void main(String[] args) {

		//G.setLoggerLevel(G.LOG_FINE);
		G.setLoggerLevel(G.LOG_INFO);
		//G.setLoggerLevel(G.LOG_DEBUG);
		// G.setLoggerShowDate(false);

		log.logln_withClassName(G.lF, "Running Weka Gaussian Naive Bayes classification for Iris Data.");
		
		WekaEnron wEnron = new WekaEnron();
		//wEnron.doWeka_UseCrossValidation();
		wEnron.doWeka_NoCrossTraining();
		wEnron.getStats();
	}


}
