package xilodyne.udacity.intro_ml.lesson1;

import java.util.List;

import mikera.arrayz.NDArray;
import xilodyne.machinelearning.classifier.GaussianNB;
import xilodyne.util.G;
import xilodyne.util.Logger;

/**
 * Java implementation of the python ClassifyNB.py from the Udacity Intro to
 * Machine Learning course.
 * 
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.1
 */

public class ClassifyNB {
	private Logger log = new Logger();

	private GaussianNB gnb;
	private double[] results;

	public void classify(NDArray features_train, double[] labels_train) {
		try {
			log.logln_withClassName(G.lF, "");
			gnb = new GaussianNB(GaussianNB.EMPTY_SAMPLES_IGNORE);
			gnb.fit(features_train, labels_train);

			// gnb.printMeanVar();
			// gnb.printAttributeValuesAndClasses();

		} catch (Exception ex) {
			ex.printStackTrace();
		}

	}

	public double[] predict(List<Double> x, List<Double> y) {
		NDArray testData = NDArray.newArray(x.size(), 2);

		for (int index = 0; index < x.size(); index++) {
			testData.set(index, 0, x.get(index));
			testData.set(index, 1, y.get(index));
		}
		// results = gnb.predict(testData);
		return gnb.predict(testData);
	}

	public double[] predict(NDArray testData) {
		// results = gnb.predict(testData);
		return gnb.predict(testData);
	}

	public void printPredict() {
		int count = 0;
		System.out.print("[");

		for (int index = 0; index < this.results.length; index++) {
			count++;
			if (count == 12) {
				count = 0;
				System.out.println();
			}
			System.out.print("\t" + String.format("%.2f", this.results[index]));
		}
		System.out.println("]");
	}

	public double getAccuracy(double[] testData, double[] resultsData) {
		return gnb.getAccuracyOfPredictedResults(testData, resultsData);
	}
}
