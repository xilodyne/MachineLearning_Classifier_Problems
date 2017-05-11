package xilodyne.udacity.intro_ml.miniproj_NB_Enron.naive_bayes;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.Array;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.Iterator;
import java.util.List;
import java.util.Queue;
import java.util.function.Function;

import mikera.arrayz.NDArray;
import weka.core.Instances;
import xilodyne.udacity.intro_ml.miniproj_NB_Enron.tools.email_preprocess;
import xilodyne.udacity.intro_ml.miniproj_NB_Enron.weka.WEKA_GNB_Enron;
import xilodyne.util.ArrayUtils;
import xilodyne.util.io.FileSplitter;
import xilodyne.util.G;
import xilodyne.util.Logger;
import xilodyne.util.io.FileUtils;
import xilodyne.util.io.jpython.PickleLoader;
import xilodyne.util.io.jpython.PythonPickleRead;
import xilodyne.util.weka.WekaARFFUtils;
import xilodyne.util.weka.WekaUtils;
import xilodyne.machinelearning.classifier.bayes.NaiveBayesClassifier_UsingTextValues;

public class nb_author_id {
	private Logger log = new Logger();
	
	NDArray featuresData = null;
	//String[] data = null;
	String[] sData = null;
	NDArray features_train = null;
	NDArray features_test = null;
	
	String[] sLabels = null;
	String[] sFeatures = null;

	
	double[] labelsData = null;
	double[] labels_train = null;
	double[] labels_test = null;
	
	Instances data = null;
	NaiveBayesClassifier_UsingTextValues nb;
	Instant startFit, endFit, startPredict, endPredict;
	String fileName, filePath;

	//split the arff file into info and data, 
	//split the data file into 5 pieces for training and testing
	//randomize file as arff file is sorted by label

	//FileSplitter.createSubFilesFromARFF(5, filePath, fileName, FileSplitter.fileExtARFF);
	//FileSplitter.createSubARFFFilesRandomize(5, filePath, fileName, FileSplitter.fileExtARFF, FileSplitter.FILE_NEEDS_RANDOMIZE);
	//using weka, get @ATTRIBUTES from info file
//	Instances data = WekaARFFUtils.wekaReadARFF(filePath + "/" +
//			fileName + "." + FileSplitter.fileExtARFF_INFO);

	
	public void loadDataFromPickleFile(){
		
		//   String label_file = "data/pickle/email_authors.20sample.pkl";
		//   String data_file = "data/pickle/word_data.20sample.pkl";
		//   this.labelsData = PickleLoader.getIntegerData(label_file);
		//   this.data = PickleLoader.getStringData(data_file);
		
	}
	
	public void loadDataFromARFF() {
//		fileName = "enron_20sample.arff";
		fileName = "enron_sample_chrissara.arff";
		filePath = "data/arff";
		data = WekaARFFUtils.wekaReadARFF(filePath + "/" +
				fileName);

		//run file splitter to sepearte data from info
		FileSplitter.createSubARFF_Shuffle(20, filePath, fileName, FileSplitter.fileExtARFF, FileSplitter.SHUFFLE);
		
	}
	
	public void ARFFdataAlreadyLoaded() {
//		fileName = "enron_20sample.arff";
		fileName = "enron_sample_chrissara.arff";
		filePath = "data/arff";
	
		log.logln(G.lF, "Reading into Weka Instance");
		data = WekaARFFUtils.wekaReadARFF(filePath + "/" +
				fileName);
		
	}
	
	public void fit_NB_Xilodyne_TextBased() {
		sLabels = WekaUtils.getLabelNames(data);
		sFeatures = WekaUtils.getFeatureNames(data);
		log.logln(G.lD, ArrayUtils.printArray(sLabels));

		List<String> labelNames = new ArrayList<String>(Arrays.asList(sLabels));
		List<String> featureNames = new ArrayList<String>(Arrays.asList(sFeatures));
		log.logln(ArrayUtils.printArray(labelNames));		
		log.logln(ArrayUtils.printArray(featureNames));

		nb = new NaiveBayesClassifier_UsingTextValues(NaiveBayesClassifier_UsingTextValues.EMPTY_SAMPLES_IGNORE);
		
		startFit = Instant.now();
		try {
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 1,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 2,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 3,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 4,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 5,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 6,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 7,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 8,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 9,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 10,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 11,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 12,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 13,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 14,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 15,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 16,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 17,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 18,1);
			fitDataFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 19,1);

		} catch (IOException e) {
			e.printStackTrace();
		}

		endFit = Instant.now();

	//	nb.printFeaturesAndLabels();
	}
	
	public void predict_NB_Xilodyne_TextBased() {

		ArrayList<String> predictedResults= null, labels = null;
		try {
//			labels = getLabelsFromFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 3,1);
			labels = getLabelsFromFile(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 20,1);
			this.startPredict = Instant.now();
//			predictedResults = this.predict(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 3,1);
			predictedResults = this.predict(filePath, fileName + "." + FileSplitter.fileExtARFF_DATA, 20,1);
		} catch (IOException e) {
			e.printStackTrace();
		}
		endPredict = Instant.now();

		try {
	} catch (Exception e) {
		e.printStackTrace();
	}

		//System.out.println("Results: " + predictedResults);
		//System.out.println("Labels:  " +labels);
		//int correctCount = ArrayUtils.getNumberOfCorrectMatches(results, labels);
		//log.logln(G.lF, "Correct match: " + correctCount + " of " + results.size());
		long trainingTime = Duration.between(startFit, endFit).toMillis();
		long predictTime = Duration.between(startPredict, endPredict).toMillis();
		System.out.println("Total lines loaded: " + nb.getFitCount());
		System.out.println("Total lines predicted: " + predictedResults.size());
		System.out.println("Predicted labels expected: " + labels.size());
		System.out.println("Training time NB: " + trainingTime + " milliseconds.");
		System.out.println("Predict time NB: " + predictTime + " milliseconds.");
		double acc = (double) ArrayUtils.getNumberOfCorrectMatches(predictedResults,  labels)/predictedResults.size();
		System.out.println("Accuracy: "  + ArrayUtils.getAccuracyOfLabels(predictedResults,  labels));

		log.logln(G.lI, "Results: " + predictedResults);
		log.logln("Labels:  " +labels);

		log.logln("Total lines loaded: " + nb.getFitCount());
		log.logln("Total lines predicted: " + predictedResults.size());
		log.logln("Predicted labels expected: " + labels.size());
		log.logln("Training time NB: " + trainingTime + " milliseconds.");
		log.logln("Predict time NB: " + predictTime + " milliseconds.");
		log.logln("Accuracy: "  + ArrayUtils.getAccuracyOfLabels(predictedResults,  labels));

	}
	
	// given file, load data minus label
	// as first value is list of strings, split the strings
	private void fitDataFile(String filePath, String fileName, 
			int fileNumber, int indexOfLabel) throws IOException {

		String combinedFileName = FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtARFF_DATA);
		String file = filePath + "/" + combinedFileName;

		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		String[] values;

		log.logln(G.lF,"Training (fitting): " + combinedFileName + ", # of lines: " + FileUtils.getLineCount(filePath, combinedFileName));

		CharSequence nullchar = "";
		//for each line, load data
		while ((line = br.readLine()) != null) {
			values = line.split(",");
			log.logln(G.lD, "Row: " + ArrayUtils.printArray(values));

			ArrayList<String> list = new ArrayList<String>();
			String label = null;
			// load the last value into the class array
			log.log("Values extracted [");
			for (int index = 0; index < values.length; index++) {
				//skip the label
				log.log_noTimestamp(values[index] + ", ");
				if (index == indexOfLabel) {
					label = values[index];
				} else {
					//remove \r\n at end of each aarf string
					String fullList = values[index];
					fullList = fullList.replaceAll("  ",  "");
					fullList = fullList.replaceAll("'",  "");
					fullList = fullList.substring(0, fullList.length()-4);

					String[] stringList = fullList.split(" ");
					for (String s : stringList) {
						if (!(s.contentEquals(nullchar)) && (s != null)) {
						list.add(s);
						}
					}
				}
			}
			log.logln_noTimestamp("]");

			log.logln(G.lD,  "Loading : " + ArrayUtils.printArray(list));
			log.logln("Label: " + label);
			Iterator<String> listLoop = list.iterator();
			while (listLoop.hasNext()) {
				nb.fit(sFeatures[0],listLoop.next(), label);
			}


		}
		br.close();
	}

	
	//assumes data has been converted to arff format, see...S

	// given file, load data minus label
	//return a list of results
	private ArrayList<String> predict(String filePath, String fileName, 
			int fileNumber, int indexOfLabel) throws IOException {

		int predictedCount = 0;
		float[] result = null;
		ArrayList<String> labels = new ArrayList<String>();

		String combinedFileName = FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtARFF_DATA);
		String file = filePath + "/" + combinedFileName;

	//	String file = filePath + "/" + FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtARFF_DATA);

		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		String[] values;

	//	log.logln(G.lI, "Predict (test): " + combinedFileName + ",");
		log.logln(G.lF,"Predict (test): " + combinedFileName + ", # of lines: " + FileUtils.getLineCount(filePath, combinedFileName));

		CharSequence nullchar = "";
		//for each line, load data
		//line = br.readLine();
		while ((line = br.readLine()) != null) {
			values = line.split(",");
			log.logln(G.lD, "Row: " + ArrayUtils.printArray(values));

			ArrayList<String> list = new ArrayList<String>();
			String label = null;
			// load the last value into the class array
			log.log("Values extracted [");
			for (int index = 0; index < values.length; index++) {
				//skip the label
				log.log_noTimestamp(values[index] + ", ");
				if (index == indexOfLabel) {
			//		label = values[index];
				} else {
					//remove \r\n at end of each aarf string
					String fullList = values[index];
					fullList = fullList.replaceAll("  ",  "");
					fullList = fullList.replaceAll("'",  "");
					fullList = fullList.substring(0, fullList.length()-4);

					String[] stringList = fullList.split(" ");
					for (String s : stringList) {
						if (!(s.contentEquals(nullchar)) && (s != null)) {
						list.add(s);
						}
					}
				}
			}
			log.logln_noTimestamp("]");

			log.logln(G.lD,  "Loading : " + ArrayUtils.printArray(list));
			log.logln("Label: " + label);
	
			//String[] wordList = new String[list.size()];
			//wordList = list.toArray(wordList);
			Hashtable<String, String> testingData_OneSet = new Hashtable<String, String>();
			for (String s : list) {
				testingData_OneSet.put(sFeatures[0], s);
			}
			String predictedLabel = nb.predict_TestingSet(testingData_OneSet);
			labels.add(predictedLabel.toLowerCase());  //lower case to match file contents
			log.logln(G.lI, "\nLooking at: " + ArrayUtils.printArray(testingData_OneSet));
			log.logln("Predicted: " + predictedLabel);

			//choose best result for line of text

	
			predictedCount++;
		}
			br.close();
		return labels;
	}

	// given file, load data minus label
	//return a list of results
	private ArrayList<String> getLabelsFromFile(String filePath, String fileName, 
			int fileNumber, int indexOfLabel) throws IOException {

//		String file = filePath + "/" + FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtARFF_DATA);

		String combinedFileName = FileSplitter.getNewFileName(fileName, fileNumber, FileSplitter.fileExtARFF_DATA);
		String file = filePath + "/" + combinedFileName;

		BufferedReader br = new BufferedReader(new FileReader(file));
		String line;
		String[] values;
		ArrayList<String> labels = new ArrayList<String>();

//		log.logln("Loading file " + file + ",");
		log.logln(G.lF,"Predict (test) labels: " + combinedFileName + ", # of lines: " + FileUtils.getLineCount(filePath, combinedFileName));


		//for each line, load data
		while ((line = br.readLine()) != null) {
			values = line.split(",");
			log.logln(G.lD, "Row: " + ArrayUtils.printArray(values));
			
			//nb.fit(new ArrayList<String>(Arrays.asList("Drew", "No", "Blue", "Short")), "Male");

			ArrayList<String> list = new ArrayList<String>();
			
			// load the last value into the class array
			log.log("Values extracted [");
			for (int index = 0; index < values.length; index++) {
				//skip the label
				log.log_noTimestamp(values[index] + ", ");
				if (index == indexOfLabel) {
					labels.add(values[index]);
				} 
			}
			log.logln_noTimestamp("]");

			log.logln("Label: " + labels.get(labels.size()-1));

		}
		br.close();
		return labels;
	}

	public void doNBAuthor(){
		log.logln_withClassName(G.LOG_FINE, "");


		//read the file
		//get the file count
		//create training data
	email_preprocess emails = new email_preprocess();
	int fileLines = emails.getEnronDataSize("./data/enron_text_sample_20.arff");

/*		NDArray features_train = NDArray.newArray(split, 2);
		NDArray features_test = NDArray.newArray(StudentMain.size - StudentMain.split, 2);
		double[] labels_train = new double[StudentMain.split];
		double[] labels_test = new double[StudentMain.size - StudentMain.split];
*/
//	featuresData = NDArray.newArray(fileLines,1);
//	labelsData = new double[fileLines];
	sData = new String[fileLines];
//	emails.loadDataSets(sData,  labelsData);
	labelsData = emails.loadLabels();
	sData = emails.loadStringLines();
	

	
	log.logln("Labels: " + ArrayUtils.printArray(labelsData));
	log.logln("Data: " + ArrayUtils.printArray(sData));

	//data = emails.readEnronData("./data/enron_text_sample_20.arff");
	//data = emails.readEnronData("../JavaJython_Test1/data/enron_text_chrissara.arff");
	
//	WekaUtils.getInstanceDetails(data);
/*	System.out.print("Convert text to integer...");
    Instances dataFiltered = CreateARFF.convertStringToNumbers(data);
    System.out.println("Done.");
    
    WEKA_GNB_Enron weka = new WEKA_GNB_Enron();
    weka.generateTestData(dataFiltered);
    
    System.out.println("no. of Chris training emails: " + weka.getClassCountFromTrainingSet(1.0));
    System.out.println("no. of Sara training emails: " + weka.getClassCountFromTrainingSet(0.0));
  */  
    
 /*   Instant start = Instant.now();
    weka.fit();
    Instant end = Instant.now();
    System.out.println(Duration.between(start, end));
 
    start = Instant.now();
    weka.getAccuracy();
    end = Instant.now();
    System.out.println(Duration.between(start, end));
   */ 
    
    
 //   System.out.println("Cross validate...");
 //   weka.crossValidate(dataFiltered);

    
   // weka.crossValidate(dataFiltered);
   // weka.getAccuracy();
  /*  //create the classifier
    String[] options = new String[1];
    options[0] = "-U";            // unpruned tree
    J48 tree = new J48();         // new instance of tree
    try {
		tree.setOptions(options);
		tree.buildClassifier(dataFiltered);   // build classifier
		   
	} catch (Exception e) {
		e.printStackTrace();
	} 
	    
    
    try {
//		emails.crossValidate(tree, dataFiltered);
		emails.crossValidate(dataFiltered);
	} catch (Exception e) {
		e.printStackTrace();
	}
	*/
    
    //System.out.println("string to numb: " + dataFiltered.toString());
    //System.out.println("filtered data: " + dataFiltered.toSummaryString());

	}
	/*
StringtoWordVector filter = new StringToWordVector();
filter.setInputFormat(TrainInstances); 
Instances FilteredTrainInstances = Filter.useFilter(TrainInstances, filter);
Instances FilteredTestInstances = Filter.useFilter(TestInstances, filter);
System.out.println(FilteredTestInstances);
*/

		
	public void doXyTest() {
		
		List <String> labelList = new ArrayList<String>(Arrays.asList("Name",">170cm","Eye","Hair"));
		List<String> classification = new ArrayList<String>(Arrays.asList("Male","Female"));

		NaiveBayesClassifier_UsingTextValues nb = new NaiveBayesClassifier_UsingTextValues(classification, labelList);
		
		nb.fit(new ArrayList<String>(Arrays.asList("Drew","No","Blue","Short")), "Male");
		nb.fit(new ArrayList<String>(Arrays.asList("Claudia","Yes","Brown","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Drew","No","Blue","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Drew","No","Blue","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Alberto","Yes","Brown","Short")), "Male");
		nb.fit(new ArrayList<String>(Arrays.asList("Karin","No","Blue","Long")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Nina","Yes","Brown","Short")), "Female");
		nb.fit(new ArrayList<String>(Arrays.asList("Sergio","Yes","Blue","Long")), "Male");

		nb.printFeaturesAndClasses(G.LOG_INFO);
		
		try {
		float[] results = nb.predictUsingFeatureNames(new ArrayList<String>(Arrays.asList("Drew","Yes","Blue","Long")));
		} catch (Exception e) {
			e.printStackTrace();
		}

	}
	public void doXyWorking2() {
		log.logln(G.LOG_INFO, "");
		//read in the data
		//create the training and testing sets
		//create bag of words
		//create td idf vector
		//classify GNB
		//show results
		Instances data, trainingWordSet, testingWordSet, trainingIntSet, testingIntSet;
		//data = weka.readEnronData("./data/enron_text_sample_20.arff");
        data = WekaARFFUtils.runWekaTextDirectoryLoader("./data/enron_sample_10");

		//convert weka data into lists
    //    WekaUtils.printInstancesLabelsAndData(data, log);
    //    WekaUtils.printInstanceDetails("enron emails", data, log);
        
		WEKA_GNB_Enron weka = new WEKA_GNB_Enron();

		weka.generateTestData(data);
		trainingWordSet = weka.getTrainingSet();
		testingWordSet = weka.getTestingSet();
        
		List<String> labelList = new ArrayList<String>(Arrays.asList("Text"));
		List<String> classification = new ArrayList<String>(Arrays.asList("sara","chris"));

		double[] sTrain_Labels = WekaUtils.loadLabelsFromWekaData(trainingWordSet, log);
		double[] sTest_Labels = WekaUtils.loadLabelsFromWekaData(testingWordSet, log);
//		System.out.println(ArrayUtils.printArray(labels));
		String[] sTrain_Data = WekaUtils.loadStringFromWekaData(trainingWordSet);
		String[] sTest_Data = WekaUtils.loadStringFromWekaData(testingWordSet);
//		System.out.println(ArrayUtils.printArray(sData));
		
		
		NaiveBayesClassifier_UsingTextValues nb = new NaiveBayesClassifier_UsingTextValues(classification, labelList);
		
		for (int index = 0; index < sTrain_Data.length; index++ ){
			String sLabel = "chris";
			if (sTrain_Labels[index] == 1 ) sLabel = "sara";
			
			String[] words = ArrayUtils.getWordsFromString(sTrain_Data[index]);
			for (String word : words) {
				nb.fit(new ArrayList<String>(Arrays.asList(word)), sLabel);
			}
	
		}

	//	nb.printFeaturesAndClasses();
		//nb.predictUsingFeatureName(labelList.indexOf("Text"), "sbaile2");
		String sResult = null;
		try {
		sResult= nb.predict(new ArrayList<String>(Arrays.asList("sbaile2")));
		} catch (Exception e) {
			e.printStackTrace();
		}

		System.out.print("Checking sbaile2... " + sResult + ": " );
		//this is a test that sbaile2 shows up as sara
		if (sResult.matches(classification.get(0)) ) {
			System.out.println("PASS");
		} else {
			System.out.println("FAIL");
		}

	//	nb.determineProbabilities();
		

		int countLabel = 0;
		for (int index = 0; index < sTest_Data.length; index++ ){
			//String[] words = ArrayUtils.getWordsFromString(sTest_Data[index]);
			//ArrayList<String> wordsTest = new ArrayList<String>();
			//wordsTest = ArrayUtils.getWordsListFromString(sTest_Data[index]);
			String[] words = ArrayUtils.getWordsFromString(sTest_Data[index]);
			//for (String word : words) {
			//	float[] results = nb.predictUsingFeatureNames(new ArrayList<String>(Arrays.asList(word)));
			//		System.out.println("results: " + results);
			//}
			
			float[] results = null;
			try {
				results = nb.predictUsingFeatureNameFromWordArray(0, words, false);
			} catch (Exception e) {
				System.out.println(e.getMessage());
				//e.printStackTrace();
			}
//					System.out.println("results: " + ArrayUtils.printArray(results));

				if (results[0] > results[1]) {
					System.out.print("Chris ");
				} else {
					System.out.print("Sara " ) ;
				}
				
				System.out.println(ArrayUtils.printArray(results));
//				System.out.println(" for " + String.valueOf(sTest_Labels[countLabel]) + " " + ArrayUtils.printArray(words));
				//System.out.println(" expected: " + String.valueOf(sTest_Labels[countLabel]) + " " + words[0] +", " + words[1]+"...");
				countLabel++;		
			
		//	System.out.println("wordsTest: " + wordsTest.toString());
		//	float[] results = nb.predictUsingFeatureNames(wordsTest);
			}


	}
	public void doXy() {
		log.logln(G.LOG_INFO, "");
		//read in the data
		//create the training and testing sets
		//create bag of words
		//create td idf vector
		//classify GNB
		//show results
		Instances data, trainingWordSet, testingWordSet, trainingIntSet, testingIntSet;
		//data = weka.readEnronData("./data/enron_text_sample_20.arff");
		log.logln("Reading data...");
	//	data = WekaARFFUtils.runWekaTextDirectoryLoader("./data/text/enron_sample_20");
     //  data = WekaARFFUtils.runWekaTextDirectoryLoader("./data/text/enron_sample_chrissara");
               
	//	WEKA_GNB_Enron weka = new WEKA_GNB_Enron();

	//	log.logln("Generating Test Data");
	//	weka.generateTestData(data);
	//	trainingWordSet = weka.getTrainingSet();
	//	testingWordSet = weka.getTestingSet();
		
		//dump sets
		//CreateARFF.writeToARFF("./data/enron-word-train-large.arff", trainingWordSet);
		//CreateARFF.writeToARFF("./data/enron-word-test-large.arff",  testingWordSet);

	//	log.logln("Writing ARFF");
		//WekaARFFUtils.writeToARFF("./data/arff/enron-word-train-large.arff", trainingWordSet);
		//WekaARFFUtils.writeToARFF("./data/arff/enron-word-test-large.arff",  testingWordSet);
   	//	WekaARFFUtils.writeToARFF("./data/arff/enron_sample_chrissara.arff",  testingWordSet);
   	//	WekaARFFUtils.writeToARFF("./data/arff/enron_sample_chrissara.arff",  data);

		//WekaARFFUtils.wekaWriteARFF("./data/arff/enron-word-train-18sample.arff", trainingWordSet);
	//	WekaARFFUtils.wekaWriteARFF("./data/arff/enron_20sample.arff",  data);
	//	log.logln("Finished Writing ARFF");
		
		   //    data = WekaARFFUtils.runWekaTextDirectoryLoader("./data/text/enron_sample_chrissara");
		//data = WekaARFFUtils.wekaReadARFF("./data/arff/enron_sample_chrissara.arff");
		data = WekaARFFUtils.wekaReadARFF("./data/arff/enron_20sample.arff");
		log.logln("Data loaded.");
		
		Instances convertedData = null;
		convertedData = WekaARFFUtils.convertToTdidfVector(data);
		//WekaUtils.printInstanceDetails("Converted Data", convertedData, log);

		//System.out.println(convertedData);
		
	//	log.logln("Writing TD-IDF ARFF");
	//	WekaARFFUtils.wekaWriteARFF("./data/arff/enron_20sample_tdidf.arff", convertedData);
		//.logln("Finished writing TD-IDF ARFF");
		//Instances convertedData = WekaARFFUtils.wekaReadARFF("./data/arff/enron_20sample_tdidf.arff");
	//	log.logln("Data loaded.");
		//WekaUtils.printInstancesLabelsAndData(convertedData, log);

			WEKA_GNB_Enron weka = new WEKA_GNB_Enron();

			weka.generateTestData(convertedData);
			trainingIntSet = weka.getTrainingSet();
			testingIntSet = weka.getTestingSet();

		//	WekaUtils.printInstanceDetails("training",  trainingIntSet, log);
			WekaUtils.printInstanceDetails("testing",  testingIntSet, log);
        
		List<String> labelList = new ArrayList<String>(Arrays.asList("Text"));
		List<String> classification = new ArrayList<String>(Arrays.asList("chris","sara"));

		double[] dTrain_Labels = WekaUtils.loadLabelsFromWekaData(trainingIntSet, log);
		double[] dTest_Labels = WekaUtils.loadLabelsFromWekaData(testingIntSet, log);
		String[] sTrain_Data = WekaUtils.loadStringFromWekaData(trainingIntSet);
		String[] sTest_Data = WekaUtils.loadStringFromWekaData(testingIntSet);		
		
		//System.out.println(ArrayUtils.printArray(sTrain_Data));
		System.out.println(ArrayUtils.printArray(sTest_Data));
		//System.out.println(ArrayUtils.printArray(dTest_Labels));
	/*	NaiveBayesClassifier nb = new NaiveBayesClassifier(classification, labelList);
		
		int found = 0;
		int found_not = 0;
		for (int index = 0; index < sTrain_Data.length; index++ ){
			String sLabel = "chris";
			if (dTrain_Labels[index] == 1 ) sLabel = "sara";
			
			String[] words = ArrayUtils.getWordsFromString(sTrain_Data[index]);
			for (String word : words) {
				nb.fit(new ArrayList<String>(Arrays.asList(word)), sLabel);
			}
	
		}

		//nb.printFeaturesAndClasses();
		String sResult = null;
		try {
		sResult= nb.predict(new ArrayList<String>(Arrays.asList("sbaile2")));
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.out.print("Checking sbaile2... " + sResult + ": " );
		//this is a test that sbaile2 shows up as sara
		if (sResult.matches(classification.get(1)) ) {
			System.out.println("PASS");
		} else {
			System.out.println("FAIL");
		}

	//	nb.determineProbabilities();
		/*
		int labelIndex = 0;
		for (int index = 0; index < sTest_Data.length; index ++) {
			int pass = 0;
			int fail = 0;

			String[] words = ArrayUtils.getWordsFromString(sTest_Data[index]);
		//	System.out.print(index +"\t");
			//get test label answer
			double labelval = dTest_Labels[labelIndex];
			String sClassLabel = classification.get((int) labelval);
			for (String word : words ) {
				//skip if word not found in dictionary

				try {
					
					String wordresult = nb.predict(new ArrayList<String>(Arrays.asList(word)));
		//			System.out.print(word +", ");
		//			System.out.println(index + "\tWord: " + word +"\tfound: " + wordresult + "\texpected: " + sClassLabel);
					if (wordresult.matches(sClassLabel)) {
						pass++;
					} else {
						fail++;
					} 
				
				} catch (Exception e) {
					//System.out.println(e.getLocalizedMessage());
				}
				
			}
	//		System.out.println();
	//		System.out.println(index + "\tlooking for: [" + labelval + "] " +sClassLabel + "\tpass/fail: " + pass + "/" + fail);

			labelIndex++;
		//	float[] results = nb.predictUsingFeatureNameFromWordArray(0, words, false);
			if (pass > fail ) {
				found++;
			} else {
				found_not++;
			}
			
			
		}

*/
				
//		System.out.println("Accuracy: "+ found / sTest_Data.length);
	}


}
