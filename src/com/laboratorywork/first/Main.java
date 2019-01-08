package com.laboratorywork.first;

import static com.laboratorywork.first.Constants.*;
import static com.laboratorywork.first.ReadFilesMNIST.ReadImagesAndLabels;

public class Main {
	public static void main(String[] args) {
		byte[] tempTrainingLabels = ReadImagesAndLabels(TRAINING_SET_LABEL_FILE, TRAINING_NUMBER_OF_IMAGES_AND_LABELS + COUNT_BYTES_OF_HEADER_LABELS); // 60000 + 8
		byte[] tempTestLabels = ReadImagesAndLabels(TEST_SET_LABEL_FILE, TEST_NUMBER_OF_IMAGES_AND_LABELS + COUNT_BYTES_OF_HEADER_LABELS); // 10000 + 8

		double[] trainingLabels = new double[TRAINING_NUMBER_OF_IMAGES_AND_LABELS];
		for (int i = 0; i < TRAINING_NUMBER_OF_IMAGES_AND_LABELS; i++)
			trainingLabels[i] = tempTrainingLabels[i + COUNT_BYTES_OF_HEADER_LABELS];
		double[] testLabels = new double[TEST_NUMBER_OF_IMAGES_AND_LABELS];
		for (int i = 0; i < TEST_NUMBER_OF_IMAGES_AND_LABELS; i++)
			testLabels[i] = tempTestLabels[i + COUNT_BYTES_OF_HEADER_LABELS];

		byte[] tempTrainingImages = ReadImagesAndLabels(TRAINING_SET_IMAGE_FILE, TRAINING_NUMBER_OF_IMAGES_AND_LABELS * IMAGE_SIZE + COUNT_BYTES_OF_HEADER_IMAGES); // 60000 * 28 * 28 + 16
		double[][] trainingImages = new double[TRAINING_NUMBER_OF_IMAGES_AND_LABELS][IMAGE_SIZE]; // 60000 * 28 * 28
		CutAndReformatsImagesArrays(tempTrainingImages, trainingImages, TRAINING_NUMBER_OF_IMAGES_AND_LABELS);

		byte[] tempTestImages = ReadImagesAndLabels(TEST_SET_IMAGE_FILE, TEST_NUMBER_OF_IMAGES_AND_LABELS * IMAGE_SIZE + COUNT_BYTES_OF_HEADER_IMAGES); // 10000 * 28 * 28 + 16
		double[][] testImages = new double[TEST_NUMBER_OF_IMAGES_AND_LABELS][IMAGE_SIZE]; // 10000 * 28 * 28
		CutAndReformatsImagesArrays(tempTestImages, testImages, TEST_NUMBER_OF_IMAGES_AND_LABELS);

		new NeuralNetwork();

		System.out.println("Training...");
		NeuralNetwork.Train(trainingImages, trainingLabels, TRAINING_NUMBER_OF_IMAGES_AND_LABELS, true);
		System.out.println("Test...");
		NeuralNetwork.Train(testImages, testLabels, TEST_NUMBER_OF_IMAGES_AND_LABELS, false);
	}

	private static void CutAndReformatsImagesArrays(byte[] tempImages, double[][] images, int numberOfImagesAndLabels) {
		for (int i = 0; i < numberOfImagesAndLabels; i++) {
			for (int j = 0; j < NUMBER_OF_ROWS_AND_COLUMNS_IMAGE; j++) {
				for (int k = 0; k < NUMBER_OF_ROWS_AND_COLUMNS_IMAGE; k++) {
					images[i][j * NUMBER_OF_ROWS_AND_COLUMNS_IMAGE + k] =
							tempImages[i * IMAGE_SIZE + j * NUMBER_OF_ROWS_AND_COLUMNS_IMAGE + k + COUNT_BYTES_OF_HEADER_IMAGES] < 0
									? (tempImages[i * IMAGE_SIZE + j * NUMBER_OF_ROWS_AND_COLUMNS_IMAGE + k + COUNT_BYTES_OF_HEADER_IMAGES] + 256) / 255.0
									: tempImages[i * IMAGE_SIZE + j * NUMBER_OF_ROWS_AND_COLUMNS_IMAGE + k + COUNT_BYTES_OF_HEADER_IMAGES] / 255.0;
				}
			}
		}
	}
}
