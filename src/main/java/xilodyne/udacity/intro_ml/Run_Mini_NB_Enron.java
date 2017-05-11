package xilodyne.udacity.intro_ml;

import xilodyne.udacity.intro_ml.miniproj_NB_Enron.naive_bayes.nb_author_id;
import xilodyne.udacity.intro_ml.miniproj_NB_Enron.weka.WekaEnron;
import xilodyne.util.G;
import xilodyne.util.Logger;

public class Run_Mini_NB_Enron {

//	private static Logger log = new Logger(G.LOG_TO_FILE);
//	private static Logger log = new Logger(G.LOG_TO_FILE_OFF);
	private static Logger log = new Logger();

	public static void main(String[] args) {

		G.setLoggerLevel(G.LOG_FINE);
		//G.setLoggerLevel(G.LOG_OFF);
		//G.setLoggerLevel(G.LOG_INFO);
		//G.setLoggerLevel(G.LOG_DEBUG);
		// G.setLoggerShowDate(false);

		log.logln_withClassName(G.lF, "Running Naive Bayes classification.");
		nb_author_id nb = new nb_author_id();
		//nb.loadDataFromARFF();
		nb.ARFFdataAlreadyLoaded();
		nb.fit_NB_Xilodyne_TextBased();
		nb.predict_NB_Xilodyne_TextBased();
	//	NBAuthID.doNBAuthor();
	//	NBAuthID.doWeka();
	//	NBAuthID.doWeka_withLoadDirectory();
	//	NBAuthID.doXy();
		
	//	WekaEnron wEnron = new WekaEnron();
		//wEnron.doWeka_UseCrossValidation();
	//	wEnron.doWeka_NoCrossTraining();
	}

}
