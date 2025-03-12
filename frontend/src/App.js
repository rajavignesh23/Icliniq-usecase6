import { useState, useEffect } from "react";
import axios from "axios";

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
    <div className="flex flex-col items-center p-6 bg-gray-100 min-h-screen">
      <h1 className="text-2xl font-bold mb-4">Disease Prediction</h1>
      {loading && <p className="text-blue-500">Fetching patient data...</p>}
      {error && <p className="text-red-500">{error}</p>}

      {isNewPatient && (
        <div className="bg-white p-4 rounded shadow w-80 mb-4">
          <h2 className="text-lg font-semibold text-blue-600">Welcome, new patient!</h2>
          <p>Start by entering your symptoms below.</p>
        </div>
      )}

      {patientData && (
        <div className="bg-white p-4 rounded shadow w-80 mb-4">
          <h2 className="text-lg font-semibold">Patient Details</h2>
          <p><strong>Username:</strong> {patientId}</p>
          <p><strong>Age:</strong> {patientData.age}</p>
          <p><strong>Gender:</strong> {patientData.gender}</p>
        </div>
      )}

      <textarea
        className="w-80 p-2 border border-gray-300 rounded mb-4"
        style={{ height: "80px" }}
        placeholder="Enter symptoms..."
        value={symptoms}
        onChange={(e) => setSymptoms(e.target.value)}
      />      

      <button
        onClick={handlePredict}
        className={`bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-all flex items-center justify-center ${
          loading ? "opacity-50 cursor-not-allowed" : ""
        }`}
        disabled={loading}
      >
        {loading ? "Predicting..." : "Predict"}
      </button>

      {results.length > 0 && (
        <div className="mt-6 w-80 bg-white p-4 rounded shadow">
          <h2 className="text-lg font-semibold mb-2">Suggested Diseases:</h2>
          <ul>
            {results.map(({ disease, probability }, index) => (
              <li key={index} className="border-b py-2">
                {disease} - <span className="text-blue-600">{(probability * 100).toFixed(2)}%</span>
              </li>
            ))}
          </ul>
        </div>
      )}

{llmResponse && typeof llmResponse === "object" && (
  <div className="mt-6 w-80 bg-white p-4 rounded shadow">
    <h2 className="text-lg font-semibold mb-2">Correlative Analysis:</h2>
    <p><strong>Final Diagnosis:</strong> {llmResponse.final_diagnosis}</p>
   
    <p><strong>Explanation:</strong> {llmResponse.explanation}</p>
    <p><strong>Suggestion:</strong> {llmResponse.suggestion}</p>
  </div>
)}



    </div>
  );
}
