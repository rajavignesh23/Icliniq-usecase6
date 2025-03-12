import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import pandas as pd
import requests
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

MODEL_PATH = "./saved_model1/model1.pth" #symtom analyser bert model path
TOKENIZER_PATH = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

MONGO_URI = "mongodb://**/" #uri with port for mongodb
DB_NAME = "" #db name 
DATA_PATH = "dataset.csv"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Saved model not found at {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

data = pd.read_csv(DATA_PATH)
label_encoder = LabelEncoder()
label_encoder.fit(data['diseases'].values)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(TOKENIZER_PATH, num_labels=len(label_encoder.classes_))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device)['model_state_dict'])
model.to(device)
model.eval()

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

llm_response = {}  

#synthetic ehr retrieval 
@app.route("/get_patient", methods=["POST"])
def get_patient():
   
    try:
        req_data = request.get_json()
        if not req_data or "patient_id" not in req_data:
            return jsonify({"error": "Missing patient_id"}), 400

        patient_id = req_data["patient_id"].strip()  
        if not patient_id:
            return jsonify({"error": "Empty patient_id"}), 400

        patient = db.patients.find_one({"patient_id": patient_id}, {"_id": 0, "age": 1, "gender": 1})
        medical_history = db.medical_history.find_one(
             {"patient_id": patient_id},
             {"_id": 0, "past_diagnoses.disease": 1, "past_diagnoses.diagnosed_year": 1}
        )

        if not patient:
            return jsonify({"error": "Patient not found"}), 404

        return jsonify({
            "age": patient.get("age", "N/A"),
            "gender": patient.get("gender", "N/A"),
            "past_diagnoses": [d.get("diagnosis", "Unknown") for d in medical_history.get("past_diagnoses", [])] if medical_history else []
        })

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    patient_id = data.get('patient_id')
    user_input = data.get('symptoms')
    print(data)
    print(patient_id)
    print(user_input)
    if not user_input:
        return jsonify({"error": "No symptoms provided"}), 400

    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1).flatten()
    topk_probs, topk_indices = torch.topk(probs, 5)
    topk_diseases = label_encoder.inverse_transform(topk_indices.cpu().numpy())

    predictions = [
        {"disease": disease, "probability": float(prob)}
        for disease, prob in zip(topk_diseases, topk_probs.cpu().numpy())
    ]

    print("\n--- Model Predictions ---")
    for p in predictions:
        print(f"Disease: {p['disease']}, Probability: {p['probability']:.2%}")
    print(patient_id)
    
    send_to_llm_async(user_input, predictions, patient_id=patient_id)

    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join()

    return jsonify({
        "predictions": predictions,
        "llm_response": llm_response.get("result", "Processing LLM response...")
    })



def send_to_llm_async(symptoms, predictions,patient_id):
    global llm_response
    thread = threading.Thread(target=lambda: llm_response.update({"result": send_to_llm(symptoms, predictions,patient_id)}))
    thread.daemon = True  
    thread.start()
    thread.join()  


import requests
import json
#simulating corelative analysis with pretrained slm
def send_to_llm(symptoms, predictions,patient_id):

    API_URL = "http://172.22.64.1:1234/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    
    
    patient = db.patients.find_one({"patient_id": patient_id}, {"_id": 0, "age": 1, "gender": 1})
    medical_history = db.medical_history.find_one(
        {"patient_id": patient_id},
        {"_id": 0, "past_diagnoses.disease": 1, "past_diagnoses.diagnosed_year": 1}
    )
    smoke = db.patients.find_one({"patient_id": patient_id}, {"_id": 0, "smoking_status": 1})
    alcohol = db.patients.find_one({"patient_id": patient_id}, {"_id": 0, "alcohol_use": 1})
    recent_visit = db.recent_visits.find_one(
        {"patient_id": patient_id},
        {"_id": 0, "symptoms": 1, "diagnosis": 1, "lab_results": 1}
    )

    #few factors/parameters from ehr for correlative analysis
    formatted_data = {
        "patient_id": patient_id,
        "age": patient.get("age", "N/A") if patient else "N/A",
        "gender": patient.get("gender", "N/A") if patient else "N/A",
        "medical_history": [
            {"disease": d["disease"], "diagnosed_year": d["diagnosed_year"]}
            for d in (medical_history.get("past_diagnoses", []) if medical_history else [])
        ],
        "lifestyle": {
            "smoking_status": smoke.get("smoking_status", "Unknown") if smoke else "Unknown",
            "alcohol_use": alcohol.get("alcohol_use", "Unknown") if alcohol else "Unknown"
        },
        "recent_visit": {
            "symptoms": recent_visit.get("symptoms", []) if recent_visit else [],
            "diagnosis": recent_visit.get("diagnosis", "Unknown") if recent_visit else "Unknown",
            "lab_results": recent_visit.get("lab_results", {}) if recent_visit else {}
        }
    }
    print(formatted_data) #
    
    system_prompt = (
        "You are a medical assistant. Your task is to verify if the predicted diseases match the given symptoms. "
        "Then, correlate with past diagnoses to suggest the most probable final diagnosis. "
        "the formatted data contains past diagnoses, smoking , alcohol use and lab results use that for correlation"
        "Provide a structured response with a two-line explanation and a one-line suggestion.\n\n"
        "Format:\n"
        "{\n"
        '  "final_diagnosis": "Most probable suggested disease and alternate disease if no suggested disease match",\n'
        '  "explanation": "Brief two-line reasoning for the diagnosis.",\n'
        '  "suggestion": "One-line recommendation for the patient."\n'
        "}\n"
    )

    
    message = {
        "symptoms": symptoms,
        "predicted_diseases": [p["disease"] for p in predictions],
        "patient_data": formatted_data
    }

    payload = {
        "model": "phi-3.1-mini-128k-instruct", #model  name
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(message, indent=4)}
        ],
        "max_tokens": 500,
        "temperature": 0.5
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code == 200:
            llm_output = response.json()["choices"][0]["message"]["content"]
            print(llm_output)  
            return llm_output  
        else:
            print(f"\nError in LLM API: {response.status_code} - {response.text}")
            return f"Error: {response.status_code} - {response.text}"

    except Exception as e:
        print(f"\n Request Failed: {e}")
        return f" Request Failed: {str(e)}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

