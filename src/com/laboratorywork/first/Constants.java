package com.laboratorywork.first;

public class Constants {
	public static final String TRAINING_SET_LABEL_FILE = "train-labels.idx1-ubyte";
	public static final String TRAINING_SET_IMAGE_FILE = "train-images.idx3-ubyte";
	public static final String TEST_SET_LABEL_FILE = "t10k-labels.idx1-ubyte";
	public static final String TEST_SET_IMAGE_FILE = "t10k-images.idx3-ubyte";

	public static final int NUMBER_OF_ROWS_AND_COLUMNS_IMAGE = 28;
	public static final int IMAGE_SIZE = 28 * 28;
	public static final int TRAINING_NUMBER_OF_IMAGES_AND_LABELS = 60000;
	public static final int TEST_NUMBER_OF_IMAGES_AND_LABELS = 10000;

	public static final int COUNT_BYTES_OF_HEADER_IMAGES = 16;
	public static final int COUNT_BYTES_OF_HEADER_LABELS = 8;

	public static final int OUTPUT = 10;
	public static final int INPUT = NUMBER_OF_ROWS_AND_COLUMNS_IMAGE * NUMBER_OF_ROWS_AND_COLUMNS_IMAGE;
	public static final double LEARNING_RATE = 0.01;
	public static final int NUMBER_OF_HIDDEN_NEURONS = 100;

	public static final int NUMBER_OF_EPOCHS = 10;
	public static final double ERROR_CROSS_ENTROPY = 0.005;
}
