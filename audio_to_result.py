import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

processor = AutoFeatureExtractor.from_pretrained("local_model")
model = AutoModelForAudioClassification.from_pretrained("local_model")

id_to_label = {0: 'Real', 1: 'AI-Generated'} 

def preprocess_and_classify_audio(file_path, target_sr=16000):
    # Step 1: Load and preprocess the audio
    audio, sample_rate = librosa.load(file_path, sr=target_sr)

    # Step 2: Preprocess the audio for model input
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")

    # Step 3: Pass the audio through the model to get predictions
    with torch.no_grad():
        predictions = model(**inputs).logits

    # Step 4: Get the predicted class (e.g., 0 or 1)
    predicted_class = torch.argmax(predictions, dim=-1).item()

    # Step 5: Map the class to a label (e.g., Real, AI-Generated)
    label = id_to_label.get(predicted_class, "Unknown")

    return predicted_class, label

if __name__ == "__main__":
    file_path = 'PranjalFolder/Dataset/linus-to-musk-DEMO.mp3'
    predicted_class, label = preprocess_and_classify_audio(file_path)
    print(f"Predicted class (number): {predicted_class}")
    print(f"Predicted label: {label}")


"""



"""



