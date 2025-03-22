# Disease Prediction Mini Project

This project is leverages a fine-tuned BERT model for predicting diseases based on patient symptoms. The system consists of three main components:

- **Frontend:** A React application that handles user authentication and displays disease predictions along with correlative analysis.
- **Backend:** A Flask API that interfaces with the machine learning model and a MongoDB database to serve patient data and prediction requests.
- **Training:** Python scripts for training and testing the BERT-based classification model on a disease dataset.


## Prerequisites

- **Node.js & npm:** For running the React frontend.
- **Python 3.7+:** For running the Flask backend and training/testing scripts.
- **MongoDB:** A running instance for storing patient records.
- **CUDA (optional):** For GPU acceleration when using PyTorch.

## Setup Instructions

### Frontend

1. Navigate to the `frontend` folder:
   ```sh
   cd frontend

Install the dependencies:
npm install

To run the development server:
npm start

 ### Backend
Navigate to the backend directory:

