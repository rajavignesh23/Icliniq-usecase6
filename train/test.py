import torch
import os
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

MODEL_PATH = "./saved_model1/model1.pth" 
TOKENIZER_PATH = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DATA_PATH = "test_split.csv"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Saved model not found at {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

test_data = pd.read_csv(DATA_PATH).dropna(subset=['symptoms'])

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
label_classes = checkpoint['label_encoder_classes']

label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(label_classes)  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(TOKENIZER_PATH, num_labels=len(label_encoder.classes_))
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"Successfully loaded model with {len(label_encoder.classes_)} labels.")




test_samples = test_data.sample(n=10, random_state=np.random.randint(0, 10000))  # Randomize on each run


correct_predictions = 0
total_samples = len(test_samples)

for _, row in test_samples.iterrows():
    user_input = row['symptoms']
    true_label = row['diseases']

    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).flatten()
    predicted_index = torch.argmax(probs).item()
    predicted_disease = label_encoder.inverse_transform([predicted_index])[0]

    print(f"Symptoms: {user_input}")
    print(f"True Disease: {true_label}")
    print(f"Predicted Disease: {predicted_disease} with Probability: {probs[predicted_index]:.4f}")
    print("-" * 50)

    if predicted_disease == true_label:
        correct_predictions += 1

accuracy = (correct_predictions / total_samples) * 100
print(f"Model Accuracy on test Samples: {accuracy:.2f}%")