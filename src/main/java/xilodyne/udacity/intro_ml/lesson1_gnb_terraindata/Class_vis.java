package xilodyne.udacity.intro_ml.lesson1_gnb_terraindata;

import java.awt.Color;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;

import mikera.arrayz.INDArray;
import mikera.arrayz.NDArray;
import xilodyne.util.jnumpy.J2NumPY;
import xilodyne.util.ArrayUtils;
import xilodyne.util.G;
import xilodyne.util.Logger;
import xilodyne.util.jmatplotlib.pyplot_ScatterPlotter;

/**
 * Java implementation of the python class_vis.py from the Udacity Intro to
 * Machine Learning course.
 * 
 * @author Austin Davis Holiday, aholiday@xilodyne.com
 * @version 0.1
 */
public class Class_vis {

	double x_min = 0.0, x_max = 1.0;
	double y_min = 0.0, y_max = 1.0;
	double h = 0.01;
	// double h = 0.1;

	private Logger log = new Logger();
	pyplot_ScatterPlotter scatterPlot;

	public void prettyPicture(ClassifyNB clf, NDArray features_test, double[] predicted) {

		double[][] XX = J2NumPY.meshgrid_getXX(J2NumPY.arange(x_min, x_max, h), J2NumPY.arange(y_min, y_max, h));
		double[][] YY = J2NumPY.meshgrid_getYY(J2NumPY.arange(x_min, x_max, h), J2NumPY.arange(y_min, y_max, h));

		log.logln_withClassName(G.lF, "Predicting frontier (backgroup display).");
		double[] boundaryDecision = clf.predict(J2NumPY._c(J2NumPY.ravel(XX), J2NumPY.ravel(YY)));
		double[][] boundaryDecisionReshaped = J2NumPY.shape1D_2_2D(boundaryDecision, XX.length, XX[0].length);

System.out.println("boundaryDecision: " + ArrayUtils.printArray(boundaryDecision));
System.out.println("reshaped: " + ArrayUtils.printArray(boundaryDecisionReshaped));
		log.logln_withClassName(G.LOG_DEBUG, "");
		// log.logln("\nPredict: " + ArrayUtils.print1DArray(result));
		log.logln("\nResultReshape: " + ArrayUtils.printArray(boundaryDecisionReshaped));

		double[] grade_sig = this.loadTestData(0, 0.0, features_test, predicted);
		double[] bumpy_sig = this.loadTestData(1, 0.0, features_test, predicted);
		double[] grade_bkg = this.loadTestData(0, 1.0, features_test, predicted);
		double[] bumpy_bkg = this.loadTestData(1, 1.0, features_test, predicted);

		log.logln("grade_sig: " + ArrayUtils.printArray(grade_sig));
		log.logln("bumpy_sig: " + ArrayUtils.printArray(bumpy_sig));
		log.logln("grade_bkg: " + ArrayUtils.printArray(grade_bkg));
		log.logln("bumpy_bkg: " + ArrayUtils.printArray(bumpy_bkg));

		scatterPlot = new pyplot_ScatterPlotter("Terrain-XilodyneGNB", "bumpiness", "grade");
		scatterPlot.addFrontier(boundaryDecisionReshaped, new Color(0, 0, 77), new Color(128, 0, 0));
		scatterPlot.scatter("fast", grade_sig, bumpy_sig, new Color(0, 0, 255));
		scatterPlot.scatter("slow", grade_bkg, bumpy_bkg, new Color(255, 0, 0));
		
	}

	private double[] loadTestData(int dimension, double classType, NDArray ndarray, double[] labels) {
		ArrayList<Double> aList = new ArrayList<Double>();

		log.logln(G.lD, "ndarray size: " + ndarray.getShape(0) + " labels size: " + labels.length);
		Iterator<INDArray> element = ndarray.iterator();

		// seems to be bug using ndarray iterator, hasNext is engaged after max
		// length
		for (int index = 0; index < labels.length; index++) {
			INDArray value = element.next();

			if (labels[index] == classType) {
				aList.add(value.get(dimension));
			}
		}

		log.logln("List count: " + aList.size());
		double[] dList = new double[aList.size()];
		for (int index = 0; index < aList.size(); index++) {
			dList[index] = aList.get(index);
		}
		return dList;
	}

	public void output_image() {
		try {
			scatterPlot.saveChartToPNG();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}