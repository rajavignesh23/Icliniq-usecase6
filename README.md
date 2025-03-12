# Medical Symptom Analysis System

## 📋 Overview

This is a smart medical system that helps predict possible diseases based on your symptoms. Think of it as a preliminary medical assistant that:
- Takes your symptoms as input
- Analyzes them using artificial intelligence
- Suggests possible conditions
- Considers your medical history for better accuracy

## 🎯 Key Features

- 🏥 Smart symptom analysis
- 📊 Disease prediction using AI
- 📱 User-friendly web interface
- 📂 Patient history tracking
- 🔒 Secure data handling

## 📁 Project Structure Explained

```
.
├── backend/                   # Server-side code
│   └── flask-route.py        # Main server file that handles requests
├── frontend/                 # User interface code
│   ├── src/                  # React application source code
│   ├── package.json         # Lists frontend app dependencies
│   └── README.md           # Frontend specific guide
├── train/                   # AI model training files
│   ├── train.py            # Script to train the AI model
│   └── test.py             # Script to test the AI model
└── README.md               # This guide you're reading
```

## 🔧 Prerequisites (What You Need Before Starting)

Before you begin, make sure you have the following installed on your computer:

1. **Python 3.8 or newer**
   - Download from: https://www.python.org/downloads/
   - To check if installed, open terminal/command prompt and type:
     ```bash
     python --version
     ```

2. **Node.js 16 or newer**
   - Download from: https://nodejs.org/
   - To verify installation:
     ```bash
     node --version
     ```

3. **MongoDB**
   - Download from: https://www.mongodb.com/try/download/community
   - Installation guides:
     - [Windows Installation Guide](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-windows/)
     - [Mac Installation Guide](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/)
     - [Linux Installation Guide](https://docs.mongodb.com/manual/administration/install-on-linux/)

4. **GPU (Optional)**
   - Having a NVIDIA GPU will make the AI model run faster
   - If you don't have one, the system will still work, just a bit slower

## 📥 Installation Guide

### Step 1: Getting the Code

1. Download this project to your computer
2. Unzip the file
3. Open terminal/command prompt
4. Navigate to the project folder:
   ```bash
   cd path/to/your/project
   ```

### Step 2: Setting Up the Backend (Server)

1. **Create a Virtual Environment** (This keeps your project dependencies separate)
   
   For Windows:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate it
   venv\Scripts\activate
   ```

   For Mac/Linux:
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate it
   source venv/bin/activate
   ```

   You'll know it's activated when you see `(venv)` at the start of your terminal line.

2. **Install Python Packages**
   ```bash
   # Make sure you're in the project root directory
   pip install -r requirements.txt
   ```
   This will install all necessary Python libraries. It might take a few minutes.

### Step 3: Setting Up the Frontend (User Interface)

1. Open a new terminal window
2. Navigate to the frontend folder:
   ```bash
   cd frontend
   ```
3. Install required packages:
   ```bash
   npm install
   ```
   This might take several minutes. Don't worry about warnings unless there are errors.

### Step 4: Database Setup

1. **Install MongoDB**
   - Follow the installation guide for your operating system (links provided in Prerequisites)
   - Make sure MongoDB is running on your computer

2. **Configure Database Connection**
   - Create a new file called `.env` in the backend folder
   - Add these lines:
     ```
     MONGO_URI=mongodb://localhost:27017
     DB_NAME=medical_system
     ```

### Step 5: Model Setup

1. **For Using Pre-trained Model:**
   - Create a folder called `saved_model1` in your project root
   - Place the model file (model1.pth) inside it
   
2. **For Training New Model:**
   - Place your dataset file (dataset.csv) in the train folder
   - The CSV file should have these columns:
     ```
     symptoms,diseases
     "fever and headache",flu
     "chest pain and shortness of breath",pneumonia
     ```
   - Run the training:
     ```bash
     cd train
     python train.py
     ```

## 🚀 Running the Application

### Starting the Backend Server

1. Open terminal in project root
2. Activate virtual environment (if not already activated):
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
3. Go to backend folder:
   ```bash
   cd backend
   ```
4. Start the server:
   ```bash
   python flask-route.py
   ```
5. You should see a message saying the server is running

### Starting the Frontend

1. Open a new terminal window
2. Go to frontend folder:
   ```bash
   cd frontend
   ```
3. Start the application:
   ```bash
   npm start
   ```
4. Your web browser should automatically open to `http://localhost:3000`

## 💻 How to Use the System

1. **Open the Application**
   - Go to `http://localhost:3000` in your web browser
   - You should see the main interface

2. **Enter Patient Information**
   - Enter the patient ID if you have one
   - Or create a new patient record

3. **Enter Symptoms**
   - Type or select symptoms in the input field
   - Be as specific as possible
   - Example: "high fever with cough and fatigue"

4. **View Results**
   - The system will analyze the symptoms
   - Show possible conditions
   - Provide recommendations

## ⚠️ Common Issues and Solutions

1. **"Could not connect to MongoDB"**
   - Make sure MongoDB is running
   - Check if the connection string in `.env` is correct
   - Try restarting MongoDB

2. **"Model not found" Error**
   - Verify the model file exists in `saved_model1/model1.pth`
   - If missing, either:
     - Download the pre-trained model, or
     - Train a new model using `train.py`

3. **"Module not found" Error**
   - Make sure you're in the virtual environment
   - Try reinstalling requirements:
     ```bash
     pip install -r requirements.txt
     ```

4. **Frontend Not Loading**
   - Check if Node.js is installed correctly
   - Try clearing npm cache:
     ```bash
     npm cache clean --force
     npm install
     ```

## 🔒 Security Best Practices

1. **Protect Your Data**
   - Never share the `.env` file
   - Keep patient data confidential
   - Use strong passwords for MongoDB

2. **Regular Updates**
   - Keep all packages updated
   - Check for security advisories
   - Update your operating system

## 🆘 Getting Help

If you encounter any issues:
1. Check the Troubleshooting section above
2. Look for error messages in the terminal
3. Make sure all prerequisites are installed
4. Verify all configuration files are set up correctly

## 📝 License

[Add your license information here]

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request.

---

Need more help? Feel free to open an issue on the project repository! 