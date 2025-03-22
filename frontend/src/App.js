import { useState, useEffect } from "react";
import axios from "axios";
import "./App.css";
export default function DiseasePredictor() {
  const [patientId] = useState(localStorage.getItem("username") || "");
  const [patientData, setPatientData] = useState(null);
  const [isNewPatient, setIsNewPatient] = useState(false);
  const [symptoms, setSymptoms] = useState("");
  const [results, setResults] = useState([]);
  const [llmResponse, setLlmResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (patientId) {
      fetchPatientData(patientId);
    }
  }, [patientId]);

  const fetchPatientData = async (patientId) => {
    setLoading(true);
    setError("");
    setPatientData(null);
    setIsNewPatient(false);
    setResults([]);
    setSymptoms("");
    setLlmResponse("");

    try {
      const response = await axios.post("http://localhost:5000/get_patient", { patient_id: patientId });
      if (response.data && Object.keys(response.data).length > 0) {
        setPatientData(response.data);
      } else {
        setIsNewPatient(true);
      }
    } catch (err) {
      setIsNewPatient(true);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!symptoms.trim()) return;
    setLoading(true);
    setResults([]);
    setLlmResponse("");
  
    console.log("DEBUG: Sending request to /predict with:", { 
      patient_id: patientId, 
      symptoms 
    });
  
    try {
      const response = await axios.post("http://localhost:5000/predict", {
        patient_id: patientId,  
        symptoms: symptoms
      });
  
      console.log("DEBUG: Response received:", response.data);
  
      setResults(response.data.predictions);
      setLlmResponse(typeof response.data.llm_response === "string" 
        ? JSON.parse(response.data.llm_response) 
        : response.data.llm_response);
      
    } catch (error) {
      console.error("Error fetching predictions", error);
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="auth-container">
      <h1 className="auth-title">Disease Prediction</h1>
      {loading && <p className="loading-text">Fetching patient data...</p>}
      {error && <p className="error-text">{error}</p>}

      {isNewPatient && (
        <div className="new-patient-box">
          <h2 className="new-patient-title">Welcome, new patient!</h2>
          <p className="new-patient-text">Start by entering your symptoms below.</p>
        </div>
      )}

      {patientData && (
        <div className="patient-details-box">
          <h2 className="patient-details-title">Patient Details</h2>
          <p className="patient-info"><strong>Username:</strong> {patientId}</p>
          <p className="patient-info"><strong>Age:</strong> {patientData.age}</p>
          <p className="patient-info"><strong>Gender:</strong> {patientData.gender}</p>
        </div>
      )}

      <textarea
        className="symptom-input"
        
        placeholder="Enter symptoms..."
        value={symptoms}
        onChange={(e) => setSymptoms(e.target.value)}
      />      

      <button
        onClick={handlePredict}
        className={`predict-button ${loading ? "disabled" : ""}`}
        disabled={loading}
      >
        {loading ? "Predicting..." : "Predict"}
      </button>

      {results.length > 0 && (
        <div className="results-box">
          <h2 className="results-title">Suggested Diseases:</h2>
          <ul>
            {results.map(({ disease, probability }, index) => (
              <li key={index} className="result-item">
                {disease} - <span className="probability-text">{(probability * 100).toFixed(2)}%</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {llmResponse && typeof llmResponse === "object" && (
        <div className="analysis-box">
          <h2 className="analysis-title">Correlative Analysis:</h2>
          <p className="analysis-text"><strong>Final Diagnosis:</strong> {llmResponse.final_diagnosis}</p>
          <p className="analysis-text"><strong>Explanation:</strong> {llmResponse.explanation}</p>
          <p className="analysis-text"><strong>Suggestion:</strong> {llmResponse.suggestion}</p>
        </div>
      )}
    </div>
  );
}
