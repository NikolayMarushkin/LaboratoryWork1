package com.laboratorywork.first;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class ReadFilesMNIST {
	public static byte[] ReadImagesAndLabels(String fileName, int lengthArray){
		byte[] imagesOrLabelsArray = new byte[lengthArray];
		try {
			imagesOrLabelsArray = Files.readAllBytes(new File(fileName).toPath());
		} catch (IOException e) {
			System.out.println("Could not open file!");
		}
		return imagesOrLabelsArray;
	}
}
