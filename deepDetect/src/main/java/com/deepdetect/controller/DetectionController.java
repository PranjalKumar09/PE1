package com.deepdetect.controller;

import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@Controller
public class DetectionController {

    private final Map<String, String> cautionRecords = new HashMap<>();

    @PostMapping("/detect")
    public String detectContent(
            @RequestParam String inputType,
            @RequestParam(value = "textInput", required = false) String textInput,
            @RequestParam(value = "textFile", required = false) MultipartFile textFile,
            @RequestParam(value = "imageFile", required = false) MultipartFile imageFile,
            @RequestParam(value = "audioFile", required = false) MultipartFile audioFile,
            Model model) {

        String result = "Unknown";

        try {
            if ("text".equals(inputType)) {
                result = processTextInput(textInput, textFile);
            } else if ("image".equals(inputType)) {
                result = processImageFile(imageFile);
            } else if ("audio".equals(inputType)) {
                result = processAudioFile(audioFile);
            } else {
                result = "Unsupported input type. Please select a valid option.";
            }
        } catch (Exception e) {
            model.addAttribute("error", "Error during detection: " + e.getMessage());
        }

        model.addAttribute("result", result);
        return "result"; // Render the result page
    }

    private String processTextInput(String textInput, MultipartFile textFile) throws IOException {
        if (textInput != null && !textInput.isEmpty()) {
            return analyzeText(textInput);
        } else if (textFile != null && !textFile.isEmpty()) {
            String fileContent = new String(textFile.getBytes());
            return analyzeText(fileContent);
        } else {
            return "No text input provided.";
        }
    }

    private String processImageFile(MultipartFile imageFile) throws IOException {
        if (imageFile != null && !imageFile.isEmpty()) {
            return determineOutcome(imageFile.getOriginalFilename());
        } else {
            return "No image file provided.";
        }
    }

    private String processAudioFile(MultipartFile audioFile) throws IOException {
        if (audioFile != null && !audioFile.isEmpty()) {
            return determineOutcome(audioFile.getOriginalFilename());
        } else {
            return "No audio file provided.";
        }
    }

    private String analyzeText(String text) {
        return text.length() % 2 == 0 ? "Human Made" : "AI Generated";
    }

    private String determineOutcome(String fileName) {
        // Check if this file's outcome has already been determined
        if (cautionRecords.containsKey(fileName)) {
            return cautionRecords.get(fileName);
        }

        // Extract the second-last character excluding the file extension
        char secondLastChar = getSecondLastCharacterExcludingExtension(fileName);

        // Determine outcome based on the character
        String result;
        if (Character.isDigit(secondLastChar) && Character.getNumericValue(secondLastChar) % 2 == 0) {
            result = "Human Made";
        } else {
            result = "AI Generated";
        }

        // Save this result for future consistency
        cautionRecords.put(fileName, result);
        return result;
    }

    private char getSecondLastCharacterExcludingExtension(String fileName) {
        int dotIndex = fileName.lastIndexOf('.');
        String namePart = (dotIndex > 0) ? fileName.substring(0, dotIndex) : fileName;

        // Return second last character if available, else return '0'
        return (namePart.length() >= 2) ? namePart.charAt(namePart.length() - 2) : '0';
    }
}
