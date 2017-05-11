package xilodyne.udacity.intro_ml.miniproj_NB_Enron.weka;

import java.time.Duration;
import java.time.Instant;

import weka.core.Instances;
import xilodyne.util.ArrayUtils;
import xilodyne.util.G;
import xilodyne.util.Logger;
import xilodyne.util.weka.WekaUtils;

public class WekaEnron {

	private Logger log = new Logger();
	Instant startFit, endFit, startPredict, endPredict = null;
	Instances data, trainingData, testingData;
	Instances  convertData, convtTrainData, convtTestData;

	public void doWeka_UseCrossValidation() {
		log.logln(G.LOG_INFO, "");
		// read in the data
		// cross validate
		// evaluate with GNB
		// show results
		Instances data, convertedData;
		WEKA_GNB_Enron weka = new WEKA_GNB_Enron();

		data = weka.readEnronData("./data/arff/enron_20sample.arff");
		// data = weka.readEnronData("./data/arff/enron_sample_chrissara.arff");

		System.out.println("\nno. of Chris training emails: " + WekaUtils.getCountFromInstancesByClass(data, 1.0));
		System.out.println("no. of Sara training emails: " + WekaUtils.getCountFromInstancesByClass(data, 0.0));

		log.logln("Convert text to int...");
		convertedData = weka.convertToTdidfVector(data);
		// WekaUtils.printInstancesLabelsAndData(convertedData, log);

		log.logln("Cross validate...");
		weka.crossValidate(convertedData);
	}
	
	public void doWeka_NoCrossTraining() {
		log.logln(G.LOG_INFO, "");
		// read in the data
		// convert to int
		// generate test data
		// evaluate with GNB
		// show results
		WEKA_GNB_Enron weka = new WEKA_GNB_Enron();

		//data = weka.readEnronData("./data/arff/enron_20sample.arff");
		data = weka.readEnronData("./data/arff/enron_sample_chrissara.arff");
		
		
		log.logln("Convert text to int...");
		convertData = weka.convertToTdidfVector(data);
		
		log.logln("Generating test data");
		weka.generateTestData(convertData);
		
		trainingData = weka.getTrainingSet();
		testingData = weka.getTestingSet();
		

		System.out.println("\nno. of Chris training emails: " + WekaUtils.getCountFromInstancesByClass(trainingData, 1.0));
		System.out.println("no. of Sara training emails: " + WekaUtils.getCountFromInstancesByClass(trainingData, 0.0));


		//convtTrainData = weka.convertToTdidfVector(trainingData);
		//convtTestData = weka.convertToTdidfVector(testingData);
		// WekaUtils.printInstancesLabelsAndData(convertedData, log);

		this.startFit = Instant.now();
		log.logln("Fit data...");
		weka.fit(trainingData);
		this.endFit = Instant.now();
		
		this.startPredict = Instant.now();
		log.logln("Test data...");
		weka.testAndPredict(testingData);
		this.endPredict = Instant.now();
	}
	
	public void getStats() {
		long trainingTime = Duration.between(startFit, endFit).toMillis();
		long predictTime = Duration.between(startPredict, endPredict).toMillis();
		System.out.println("Total lines training: " + WekaUtils.getCountFromInstances(trainingData));
		System.out.println("Total lines predicted: " + WekaUtils.getCountFromInstances(testingData));
		System.out.println("Training time NB: " + trainingTime + " milliseconds.");
		System.out.println("Predict time NB: " + predictTime + " milliseconds.");
//		double acc = (double) ArrayUtils.getNumberOfCorrectMatches(predictResults,  labels)/predictResults.size();
//		System.out.println("Accuracy: "  + ArrayUtils.getAccuracyOfLabels(predictResults,  labels));

	}

}
