package com.laboratorywork.first;

import java.util.Collections;
import java.util.Random;

import static com.laboratorywork.first.Constants.*;
import static java.lang.Math.exp;
import static java.lang.Math.log;

public class NeuralNetwork {

	static Random random = new Random(System.currentTimeMillis());

	static double[] inputArray = new double[INPUT];
	static double[] outputArray = new double[OUTPUT];
	static double[] hideOutput = new double[NUMBER_OF_HIDDEN_NEURONS];
	static double[] deltaWeightOnOutput = new double[OUTPUT];
	static double[] deltaWeightOnHide = new double[NUMBER_OF_HIDDEN_NEURONS];
	static double[] gradientHide = new double[NUMBER_OF_HIDDEN_NEURONS];
	static double[] gradientOutput = new double[OUTPUT];
	static double[][] hideWeights = new double[INPUT][NUMBER_OF_HIDDEN_NEURONS];
	static double[][] outputWeights = new double[NUMBER_OF_HIDDEN_NEURONS][OUTPUT];

	public NeuralNetwork() {
		for (int i = 0; i < OUTPUT; i++) {
			deltaWeightOnOutput[i] = random.nextDouble() - random.nextInt(2); //(-1.0, 1.0)
			gradientOutput[i] = 0;
		}
		for (int i = 0; i < NUMBER_OF_HIDDEN_NEURONS; i++) {
			deltaWeightOnHide[i] = random.nextDouble() - random.nextInt(2); //(-1.0, 1.0)
			gradientHide[i] = 0;
		}
		for (int i = 0; i < INPUT; i++)
			for (int j = 0; j < NUMBER_OF_HIDDEN_NEURONS; j++) {
				hideWeights[i][j] = random.nextDouble() - random.nextInt(2); //(-1.0, 1.0)
			}

		for (int i = 0; i < NUMBER_OF_HIDDEN_NEURONS; i++)
			for (int j = 0; j < OUTPUT; j++) {
				outputWeights[i][j] = random.nextDouble() - random.nextInt(2); //(-1.0, 1.0)
			}
	}

	public static void Train(double[][] images, double[] labels, int countImages, boolean isTrain) {
		double[] expOutput = new double[OUTPUT];
		int countEpochs = 1;
		while (countEpochs <= NUMBER_OF_EPOCHS) {
			double countCorrectAnswers = 0.0;
			MixImages(images, labels, countImages);
			System.out.println("Number of epochs: " + countEpochs);

			for (int i = 0; i < countImages; i++) {
				for (int j = 0; j < INPUT; j++) inputArray[j] = images[i][j];
				for (int j = 0; j < OUTPUT; j++) {
					expOutput[j] = 0.0;
					if (j == labels[i]) expOutput[j] = 1.0;
				}
				outputArray = CalculateOutput(CalculateHideOutput(inputArray));
				if (expOutput[IndexOfMaxElement(outputArray)] == 1.0) countCorrectAnswers++;
				if (isTrain) {
					CalculateGradient(expOutput);
					CalculateWeights(gradientOutput, gradientHide);
					CalculateDelta(gradientOutput, gradientHide);
				} else {
					countEpochs = NUMBER_OF_EPOCHS;
				}
			}
			double crossEntropy = CalculateCrossEntropy(images, labels, countImages);
			System.out.println("Cross Entropy: " + crossEntropy);
			double accuracy = countCorrectAnswers / countImages;
			System.out.println("Accuracy: " + accuracy);
			countEpochs++;
			if((crossEntropy <= ERROR_CROSS_ENTROPY) || (1 - accuracy <= ERROR_CROSS_ENTROPY)) break;
		}
	}

	private static double CalculateCrossEntropy(double[][] images, double[] labels, int countImages) {
		double sum = 0.0;
		double[] x = new double[INPUT];
		double[] p = new double[OUTPUT];
		double[] q;
		for (int i = 0; i < countImages; i++) {
			for (int j = 0; j < INPUT; j++) x[j] = images[i][j];
			for (int j = 0; j < OUTPUT; j++) {
				p[j] = 0.0;
				if (j == labels[i]) p[j] = 1.0;
			}
			q = CalculateOutput(CalculateHideOutput(x));
			for (int j = 0; j < OUTPUT; j++) sum += p[j] * log(q[j]);
		}
		return -sum / countImages;
	}

	private static void CalculateDelta(double[] gradientOutput, double[] gradientHide) {
		for (int i = 0; i < OUTPUT; i++) deltaWeightOnOutput[i] += LEARNING_RATE * gradientOutput[i];
		for (int i = 0; i < NUMBER_OF_HIDDEN_NEURONS; i++) deltaWeightOnHide[i] += LEARNING_RATE * gradientHide[i];
	}

	private static void CalculateWeights(double[] gradientOutput, double[] gradientHide) {
		for (int i = 0; i < NUMBER_OF_HIDDEN_NEURONS; i++)
			for (int j = 0; j < OUTPUT; j++) outputWeights[i][j] += LEARNING_RATE * gradientOutput[j] * hideOutput[i];
		for (int i = 0; i < INPUT; i++)
			for (int j = 0; j < NUMBER_OF_HIDDEN_NEURONS; j++)
				hideWeights[i][j] += LEARNING_RATE * gradientHide[j] * inputArray[i];
	}

	private static void CalculateGradient(double[] expOutput) {
		for (int i = 0; i < OUTPUT; i++) gradientOutput[i] = (expOutput[i] - outputArray[i]);
		double sum = 0.0;
		for (int i = 0; i < NUMBER_OF_HIDDEN_NEURONS; i++) {
			for (int j = 0; j < OUTPUT; j++)
				sum += gradientOutput[j] * outputWeights[i][j];
			gradientHide[i] = sum * hideOutput[i] * (1 - hideOutput[i]);
		}
	}

	private static int IndexOfMaxElement(double[] outputArray) {
		double maxElement = outputArray[0];
		int indexElement = 0;
		for (int i = 1; i < OUTPUT; i++)
			if (maxElement < outputArray[i]) {
				maxElement = outputArray[i]; indexElement = i;
			}
		return indexElement;
	}

	private static double[] CalculateHideOutput(double[] x) {
		double[] sum = new double[NUMBER_OF_HIDDEN_NEURONS];
		for (int i = 0; i < NUMBER_OF_HIDDEN_NEURONS; i++) sum[i] = 0.0;
		for (int i = 0; i < NUMBER_OF_HIDDEN_NEURONS; i++)
			for (int j = 0; j < INPUT; j++) sum[i] += x[j] * hideWeights[j][i];
		for (int i = 0; i < NUMBER_OF_HIDDEN_NEURONS; i++) sum[i] += deltaWeightOnHide[i];
		for (int i = 0; i < NUMBER_OF_HIDDEN_NEURONS; i++) hideOutput[i] = 1.0 / (1.0 + exp(-sum[i]));
		return hideOutput;
	}

	private static double[] CalculateOutput(double[] hideOutput){
		double[] sum = new double[OUTPUT];
		for (int i = 0; i < OUTPUT; i++) sum[i] = 0.0;
		for (int i = 0; i < OUTPUT; i++)
			for (int j = 0; j < NUMBER_OF_HIDDEN_NEURONS; j++) sum[i] += hideOutput[j] * outputWeights[j][i];
		for (int i = 0; i < OUTPUT; i++) sum[i] += deltaWeightOnOutput[i];
		outputArray = Softmax(sum);
		return outputArray;
	}

	private static double[] Softmax(double[] sum) {
		double[] expInput = new double[OUTPUT];
		double[] softmax = new double[OUTPUT];
		double sumExp = 0.0;
		for (int i = 0; i < OUTPUT; i++) expInput[i] = exp(sum[i]);
		for (int i = 0; i < OUTPUT; i++) sumExp += expInput[i];
		for (int i = 0; i < OUTPUT; i++) softmax[i] = expInput[i] / sumExp;
		return softmax;
	}

	private static void MixImages(double[][] images, double[] labels, int countImages) {
		for (int i = 0; i < countImages; i++) {
			//swap Images
			int iImages1 = random.nextInt(countImages);
			int jImages1 = random.nextInt(IMAGE_SIZE);
			int iImages2 = random.nextInt(countImages);
			int jImages2 = random.nextInt(IMAGE_SIZE);
			double tempImages = images[iImages1][jImages1];
			images[iImages1][jImages1] = images[iImages2][jImages2];
			images[iImages2][jImages2] = tempImages;
			//swap Labels
			int iLabels1 = random.nextInt(countImages);
			int iLabels2 = random.nextInt(countImages);
			double tempLabels = labels[iLabels1];
			labels[iLabels1] = labels[iImages2];
			labels[iLabels2] = tempLabels;
		}
	}

}
