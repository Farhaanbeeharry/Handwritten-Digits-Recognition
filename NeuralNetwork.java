import coursework2.Calculations;
import java.io.FileNotFoundException;

public class NeuralNetwork {
	
	private int[][] trainingData;
	private int[][] testData;
	private int inputSize = 64, hiddenSize = 256, outputSize = 16;
	private double learningRate = 0.1;
	private double weightsLayer1[][];
	private double weightsLayer2[][];
	private double activationInput[];
	private double activationHidden[];
	private double activationOutput[];
	private double trainingAccuracy;
	
	public NeuralNetwork(int[][] trainData, int[][] testData) {
		this.trainingData = trainData;
		this.testData = testData;
		
		this.weightsLayer1 = new double[this.inputSize][this.hiddenSize];
		this.weightsLayer2 = new double[this.hiddenSize][this.outputSize];
		
		this.activationInput = new double[inputSize];
		this.activationHidden = new double[hiddenSize];
		this.activationOutput = new double[outputSize];
		
		this.activationHidden[this.hiddenSize - 1]   = 1.0; // Bias units
		this.activationInput[this.inputSize  - 1]   = 1.0;
		
		this.trainingAccuracy = 0.0;
	}
	
	public void train(int epochs) {
		
		double[][] inputs = new double[this.trainingData.length][inputSize];
		double[][] outputs = new double[this.trainingData.length][outputSize]; 
		
		for(int x = 0; x < this.trainingData.length; x++) {
			for(int y = 0; y < this.inputSize-1; y++) {
				inputs[x][y] = this.trainingData[x][y];
			}
			inputs[x][this.inputSize-1] = 1;
		}
		
		for(int m = 0; m < this.trainingData.length; m++) {
			for(int n = 0; n < outputSize; n++) {
				if(n == trainingData[m][64]) {
					outputs[m][n] = 1;
				}
				else {
					outputs[m][n] = 0;
				}
			}
		}
		
		int total = 0;
		initialiseWeights();
		for (int e = 0; e < epochs; e++) {
			for (int i = 0; i < inputs.length; i++) {
				ActivateNeurons(inputs[i]);
				double[] errors = new double[outputSize];
				for (int k = 0; k < outputSize; k++)
					errors[k] = outputs[i][k] - activationOutput[k]; // Y - YHAT
				AdjustWeights(errors);
				if(e == epochs-1) {
					int actualOutput = getMaxIndex(activationOutput);
					int expectedOutput = getMaxIndex(outputs[i]);
					if(expectedOutput == actualOutput) {
						total++;
					}
				}
			}
		}
		this.trainingAccuracy = (total/2810.0)*100.0;
	}
	
	public void test() {
		
		train(660);
		
		double[][] inputs = new double[this.testData.length][inputSize];
		double[][] outputs = new double[this.testData.length][outputSize]; 
		
		for(int x = 0; x < this.testData.length; x++) {
			for(int y = 0; y < this.inputSize-1; y++) {
				inputs[x][y] = this.testData[x][y];
			}
		}
		
		for(int m = 0; m < this.testData.length; m++) {
			for(int n = 0; n < outputSize; n++) {
				if(n == testData[m][64]) {
					outputs[m][n] = 1;
				}
				else {
					outputs[m][n] = 0;
				}
			}
		}
		
		int total = 0;
		for(int i = 0; i < inputs.length; i++) {
			ActivateNeurons(inputs[i]);
			int actualOutput = getMaxIndex(activationOutput);
			int expectedOutput = getMaxIndex(outputs[i]);
			if(expectedOutput == actualOutput) {
				total++;
			}
			System.out.println("Predicted digit: "+ actualOutput);
		}
		
		System.out.println("----------- Final Results ------------");
		System.out.println("Train Accuracy for Neural Network: "+Calculations.roundOff2DP(this.trainingAccuracy));
      		
        
                double testAccuracy = (total/2810.0)*100.0;
		System.out.println("Test Accuracy for Neural Network: "+Calculations.roundOff2DP(testAccuracy));
	
	}
	

	public void initialiseWeights() {
		for (int i = 0; i < inputSize; i++)
			for (int j = 0; j < hiddenSize; j++)
				weightsLayer1[i][j] = -1.0 + (1.0 - (-1.0)) * Math.random();
		for (int j = 0; j < hiddenSize; j++)
			for (int k = 0; k < outputSize; k++)
				weightsLayer2[j][k] = -1.0 + (1.0 - (-1.0)) * Math.random();
	}

	public void ActivateNeurons(double[] inputs) {
		// store values for activations of input neurons
		for (int x = 0; x < inputSize - 1; x++)
			activationInput[x] = inputs[x];

		// store values for activations of hidden layer neurons
		for (int j = 0; j < hiddenSize - 1; j++) {
			activationHidden[j] = 0.0;
			for (int i = 0; i < inputSize; i++)
				activationHidden[j] += weightsLayer1[i][j] * activationInput[i];
			activationHidden[j] = sigmoid(activationHidden[j]);
		}

		// store values for activations of output layer neurons
		for (int k = 0; k < outputSize; k++) {
			activationOutput[k] = 0.0;
			for (int j = 0; j < hiddenSize; j++)
				activationOutput[k] += activationHidden[j] * weightsLayer2[j][k];
			activationOutput[k] = sigmoid(activationOutput[k]);
		}
	}
	
	public void AdjustWeights(double errors[]) {
		// get the output's delta values with the product of their derivatives and errors
		double[] outputDelta = new double[outputSize];
		for (int k = 0; k < outputSize; k++)
			outputDelta[k] = getDerivative(activationOutput[k]) * errors[k];
		
		// get the hidden layer's delta values
		double[] hiddenDelta = new double[hiddenSize];
		for (int j = 0; j < hiddenSize; j++)
			for (int k = 0; k < outputSize; k++)
				hiddenDelta[j] += getDerivative(activationHidden[j]) * outputDelta[k] * weightsLayer2[j][k];

		// Adjust weights from the first layer (Input to hidden)
		for (int i = 0; i < inputSize; i++)
			for (int j = 0; j < hiddenSize; j++)
					weightsLayer1[i][j] += learningRate * hiddenDelta[j] * activationInput[i];

		// Adjust weights from the first layer (Hidden to output)
		for (int j = 0; j < hiddenSize; j++)
			for (int k = 0; k < outputSize; k++)
				weightsLayer2[j][k] += learningRate * outputDelta[k] * activationHidden[j];
	}

	private int getMaxIndex(double[] outputs) {
		double max = outputs[0];
		int index = 0;
		for(int x = 0; x < this.outputSize; x++) {
			if(outputs[x] > max) {
				index = x;
				max = outputs[x];
			}
		}
		return index;
	}

	private double sigmoid(double actA) {
		return 1./(1 + Math.exp(-actA));
	}
	
	private double getDerivative(double y) {
		return y * (1 - y);
	}
	
	}
