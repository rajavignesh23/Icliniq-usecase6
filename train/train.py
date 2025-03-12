import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

class SymptomDiseaseDataset(Dataset):
    def __init__(self, data, tokenizer, label_encoder, max_length=128):
        data = data.dropna(subset=['symptoms'])  
        data.loc[:, 'symptoms'] = data['symptoms'].astype(str)  
        self.labels = label_encoder.transform(data['diseases'].values)
        self.texts = data['symptoms'].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encodings = self.tokenizer(text, padding='max_length', max_length=self.max_length, truncation=True, return_tensors='pt')
        return {
            'input_ids': encodings['input_ids'].flatten(),
            'attention_mask': encodings['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

data = pd.read_csv('dataset.csv')


train_data, eval_data = train_test_split(data, test_size=0.2, random_state=42)

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
label_encoder = LabelEncoder()
label_encoder.fit(data['diseases'].values)

train_dataset = SymptomDiseaseDataset(train_data, tokenizer, label_encoder)
eval_dataset = SymptomDiseaseDataset(eval_data, tokenizer, label_encoder)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Increased batch size
eval_loader = DataLoader(eval_dataset, batch_size=16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    num_labels=len(label_encoder.classes_)
)
model.to(device)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,  # Reduced number of epochs 
    per_device_train_batch_size=16,  # Increased batch size
    per_device_eval_batch_size=16,  # Increased batch size
    warmup_steps=0,  # Disable warm-up
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,  # Reduced logging frequency
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch
    save_strategy="epoch",  # Save model at the end of each epoch
    load_best_model_at_end=True,  # Load the best model at the end
    learning_rate=5e-5,  # Fixed learning rate
    lr_scheduler_type='constant',  # No learning rate decay
    fp16=True,  # Enable mixed precision training
    gradient_accumulation_steps=2,  # Simulate a larger batch size
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

trainer.save_model("./saved_model")
torch.save({
    'model_state_dict': model.state_dict(),
    'label_encoder_classes': label_encoder.classes_
}, "./saved_model1/model1.pth")

print("Model successfully saved") # trained model weight&bias -> symptom anlysis

def suggest_diseases(user_input, model, tokenizer, label_encoder, device, top_k=5):
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).flatten()

    topk_probs, topk_indices = torch.topk(probs, top_k)
    topk_diseases = label_encoder.inverse_transform(topk_indices.cpu().numpy())

    print("Suggested diseases based on symptoms:")
    for disease, prob in zip(topk_diseases, topk_probs.cpu().numpy()):
        print(f"{disease}: {prob:.4f}")


user_symptoms = input()#"I have a rash and joint pain"
suggest_diseases(user_symptoms, model, tokenizer, label_encoder, device)
