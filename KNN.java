import coursework2.Calculations;
import java.util.ArrayList;

public class KNN {

	private int[][] trainingData;
	private int[][] testData;
	private int K;
	private double[][] neighbors;

	public KNN(int[][] trainData, int[][] testData, int k) {
		this.trainingData = trainData;
		this.testData = testData;
		this.K = k;
	}

	public void trainData(int row, int[][] data) {

		for (int x = 0; x < this.trainingData.length; x++) {
			this.neighbors[x][1] = this.trainingData[x][64];
			double distance = getEuclideanDistance(row, x, data, this.trainingData);
			this.neighbors[x][0] = distance;
		}

	}

	/**
	 * Training a data set using KNN algorithm and measuring its accuracy
	 */
	public void train() {
		// initialize neighbors
		this.neighbors = new double[this.trainingData.length][2]; // array with distance and class of row

		int count = 0;
		// train system with data
		for (int row = 0; row < this.trainingData.length; row++) {
		
			for (int i = 0; i < neighbors.length; i++) {
				this.neighbors[i][0] = 10000;
				this.neighbors[i][1] = -1;
			}

			int[] nearestNeighbors = new int[this.K];

			for (int x = 0; x < this.trainingData.length; x++) {
				if (x != row) {
					this.neighbors[x][1] = this.trainingData[x][64];
					double distance = getEuclideanDistance(row, x, this.trainingData, this.trainingData);
					this.neighbors[x][0] = distance;
				}
			}

			this.neighbors = sortDistances(this.neighbors);

			for (int n = 0; n < this.K; n++) {
				nearestNeighbors[n] = (int) this.neighbors[n][1];
			}

			if (getMode(nearestNeighbors) == this.trainingData[row][64]) {
				count++;
			}

		}

		double trainAccuracy = count / 2810.0 * 100.0;
		System.out.println("KNN Traning Accuracy: " + Calculations.roundOff2DP(trainAccuracy));
	}


	public void test() {
		// initialize neighbors
		this.neighbors = new double[this.testData.length][2]; // array with distance and class of row

		int count = 0;
		// train system with data
		for (int row = 0; row < this.testData.length; row++) {
		
			for (int i = 0; i < neighbors.length; i++) {
				this.neighbors[i][0] = 10000;
				this.neighbors[i][1] = -1;
			}

			int[] nearestNeighbors = new int[this.K];

			trainData(row, this.testData);

			this.neighbors = sortDistances(this.neighbors);

			for (int n = 0; n < this.K; n++) {
				nearestNeighbors[n] = (int) this.neighbors[n][1];
			}

			if (getMode(nearestNeighbors) == this.testData[row][64]) {
				count++;
			}
			System.out.println("Predicted Digit: " + getMode(nearestNeighbors));
		
		}

		System.out.println("----------- Final Results ------------");
		double testAccuracy = count / 2810.0 * 100.0;
		System.out.println("KNN Test Accuracy: " + Calculations.roundOff2DP(testAccuracy));
	}

	public int getMode(int[] list) {
		int mode = 0, maxCount = 0, i, j;

		for (i = 0; i < list.length; ++i) {
			int count = 0;
			for (j = 0; j < list.length; ++j) {
				if (list[j] == list[i])
					++count;
			}

			if (count > maxCount) {
				maxCount = count;
				mode = list[i];
			}
		}
		return mode;
	}

	public double[][] sortDistances(double[][] list) {
		double temp1, temp2;
		for (int i = 0; i < list.length; i++) {
			for (int j = i + 1; j < list.length; j++) {
				if (list[i][0] > list[j][0]) {
					temp1 = list[i][0];
					list[i][0] = list[j][0];
					list[j][0] = temp1;

					temp2 = list[i][1];
					list[i][1] = list[j][1];
					list[j][1] = temp2;
				}
			}
		}
		return list;
	}

	public double getEuclideanDistance(int rowNum, int otherRowNum, int[][] test, int[][] train) {
		double sum = 0;
		for (int x = 0; x < 64; x++) {
			sum += Math.pow((test[rowNum][x] - train[otherRowNum][x]), 2);
		}
		return Math.sqrt(sum);
	}

}
