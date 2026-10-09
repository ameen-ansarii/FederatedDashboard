# FederatedDashboard: Privacy-Preserving Federated Learning for Student Dropout Prediction

A privacy-preserving Learning Analytics system that predicts student academic attrition using Federated Learning (FedProx & FedAvg) and Explainable AI (SHAP attributions) on the Open University Learning Analytics Dataset (OULAD).

## Key Features

- **Decentralized Campus Training**: Colleges train models locally on institutional student data without transmitting raw records to central servers.
- **Drift Mitigation with FedProx**: Mitigates client heterogeneity across colleges using proximal regularization penalty ($\mu$).
- **Explainable AI (XAI)**: Identifies primary risk drivers (portal clicks, assignment grades, course repeat history) with actionable intervention recommendations.
- **Real-Time Cohort Risk Evaluation**: Institutional batch CSV evaluator for predicting and ranking entire student cohorts locally with zero data egress.
- **Interactive Single-Student Sandbox**: What-if simulation tool for academic advisors to evaluate parameter changes in real time.

## Architecture

- **Backend**: FastAPI with PyTorch tensor engine (`backend_lite.py` / `app/`)
- **Frontend**: React 18, Vite, TypeScript, Tailwind CSS, Framer Motion
- **Dataset**: Open University Learning Analytics Dataset (OULAD)

## Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 2. Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the PyTorch backend server
python backend_lite.py
```
Backend will start on `http://127.0.0.1:8000`.

### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```
Frontend will be accessible at `http://localhost:5173`.
