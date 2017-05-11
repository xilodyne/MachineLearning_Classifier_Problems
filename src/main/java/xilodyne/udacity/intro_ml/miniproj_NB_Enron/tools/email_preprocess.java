package xilodyne.udacity.intro_ml.miniproj_NB_Enron.tools;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import mikera.arrayz.NDArray;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;
import xilodyne.util.ArrayUtils;
import xilodyne.util.G;
import xilodyne.util.Logger;
import xilodyne.util.io.jpython.PythonPickleRead;
import xilodyne.util.weka.WekaARFFUtils;
import xilodyne.util.weka.WekaUtils;

//would be much easier to keep everything in a Weka instance but
//the goal is to match the udacity python format of feature_train/test and
//labels_train/test
public class email_preprocess {

	private Logger log = new Logger();

	Instances wekaData = null;
	
	public email_preprocess () {
		log.logln_withClassName(G.LOG_INFO, "");

	}
	
	public void preprocess() {
		//read the files
		double[] labels = PythonPickleRead.readPickleFileFeatures("allnumber.pkl", "./data" );
		ArrayUtils.printArray(labels);
	}
	
	//read the file and get the size
	public int getEnronDataSize(String filename) {
		this.readEnronData(filename);
		return this.wekaData.numInstances();
	}
	
	public void readEnronData(String filename) {
		BufferedReader reader;
		try {
			reader = new BufferedReader(new FileReader(filename));
			 ArffReader arff = new ArffReader(reader);
					 wekaData = arff.getData();
					 wekaData.setClassIndex(wekaData.numAttributes() - 1);
			//wekaData = new Instances(reader);
			reader.close();
			// setting class attribute
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		wekaData.setClassIndex(wekaData.numAttributes() - 1);
	}
	
	public void loadDataSets(String[] sData, double[] labels) {
		WekaUtils.getInstanceDetails(wekaData);
		labels = WekaUtils.loadLabelsFromWekaData(wekaData, log);
		sData = WekaUtils.loadLinesFromWekaData(wekaData);
	}
	public double[] loadLabels() {
		WekaUtils.getInstanceDetails(wekaData);
		return WekaUtils.loadLabelsFromWekaData(wekaData, log);
	}
	
	public String[] loadStringLines() {
		return WekaUtils.loadLinesFromWekaData(wekaData);
	}
				
/*		System.out.print("Convert text to integer...");
	    Instances dataFiltered = CreateARFF.convertStringToNumbers(data);
	    System.out.println("Done.");
	    
	    WEKA_GNB_Enron weka = new WEKA_GNB_Enron();
	    weka.generateTestData(dataFiltered);
	*/ //   }
	
		//convert the instances

	public Instances readEnronDataToNDArray(String filename) {
		BufferedReader reader;
		Instances data = null;
		try {
			reader = new BufferedReader(new FileReader(filename));
			data = new Instances(reader);
			reader.close();
			// setting class attribute
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public void crossValidate(J48 tree, Instances data) throws Exception {
		System.out.println("Data: " + data.toString());
		 Evaluation eval = new Evaluation(data);
		 eval.crossValidateModel(tree, data, 10, new Random(1));
		 System.out.println("Eval: " + eval.toSummaryString());
			System.out.println("Data: " + data.toString());		 
	}
	
	public void crossValidate(Instances data) throws Exception {
//		AbstractClassifier classifier = ( AbstractClassifier ) Class.forName(
//	            "weka.classifiers.functions.LibSVM" ).newInstance();
		AbstractClassifier classifier = ( AbstractClassifier ) Class.forName(
	            "weka.classifiers.bayes.NaiveBayes" ).newInstance();

		System.out.println("Data: " + data.toString());
		 Evaluation eval = new Evaluation(data);
		 eval.crossValidateModel(classifier, data, 10, new Random(1));
		 System.out.println("Eval: " + eval.toSummaryString());
			System.out.println("Data: " + data.toString());		 
	}
}
