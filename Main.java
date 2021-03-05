import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;

public class Main {

	public static void main(String[] args) {
		
		// TODO Auto-generated method stub

        int[][] trainingData = new int[2810][65];
        int[][] testData = new int[2810][65];
        
        String TrainingCsvFile = "data/cw2DataSet1.csv";
		String TestCsvFile = "data/cw2DataSet2.csv";
        
		trainingData = loadData(TrainingCsvFile, trainingData);
		testData = loadData(TestCsvFile, testData);
		

		System.out.print("Which algorithm you want to run?:\n1: K-Nearest Neighbour \n2: Neural Network (Multi-layer Perceptron)\n");
		System.out.println("\nEnter the number corresponding to your choice: ");
		
		// This part will read the value entered by user
		String number = null;
		BufferedReader bufferedReader = new BufferedReader(new InputStreamReader(System.in));
		try {
			number = bufferedReader.readLine();
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			System.out.println("Invalid input! \nPlease select between 1 and 2.");
//					e.printStackTrace();
		}
		
		switch(number) {
		  case "1":
			KNN solution1 = new KNN(trainingData, testData, 4);
			solution1.test();
			solution1.train();
		    break;
		  case "2":
			  NeuralNetwork solution2 = new NeuralNetwork(trainingData, testData);
			  solution2.test();
		    break;
		  
		  default:
			System.out.println("The value you enterred was not a number from the list above \nBy default, the first option(KNN) will be selected...");
			KNN solution3 = new KNN(trainingData, testData, 4);
			solution3.test();
			solution3.train();
		}

	}
	
	public static int[][] loadData(String filename, int[][] data){
		
		String line = "";
        String cvsSplitBy = ",";
        
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {

        	int c = 0;
            while ((line = br.readLine()) != null) {
            	
                // use comma as separator
                String[] digits = line.split(cvsSplitBy);
                for(int i = 0; i < digits.length; i++) {
                	data[c][i] = Integer.parseInt(digits[i]);
                }
                c++;
            }
            
            return data;

        } catch (IOException e) {
            e.printStackTrace();
        }
		return null;
	}

}
