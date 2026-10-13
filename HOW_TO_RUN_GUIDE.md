# Professor & Evaluator Guide: How to Run the Federated Education System (FL-EdSys)

This guide provides step-by-step instructions for running and evaluating the **Federated Learning-Based Education System (FL-EdSys)** for student dropout prediction and explainable learning analytics.

---

## 1. What Things Need to Be There (Prerequisites & File Checklist)

### System Prerequisites
To run the full interactive system, only **one single prerequisite** is required on the machine:
* **Python 3.10, 3.11, or 3.12** (available free from [python.org](https://www.python.org/downloads/))
  * *Important during Python install*: Ensure the checkbox **"Add Python to PATH"** is ticked.

> **Zero Extra Tooling Needed:** Node.js, npm, Docker, and external database servers are **not required** to evaluate the demo. The production React dashboard is already compiled into `frontend/dist/` and is served directly by the PyTorch backend engine.

---

### File Checklist (Verify these are in the project folder)

| File / Folder | Purpose |
| :--- | :--- |
| `START_DEMO.bat` | **One-click launcher for Windows** (auto-checks packages & opens browser) |
| `start_demo.sh` | **One-command launcher for macOS / Linux** |
| `backend_lite.py` | Unified FastAPI server + PyTorch AI Inference & FL engine |
| `frontend/dist/` | Pre-built production frontend (HTML5, Tailwind CSS, React) |
| `storage_data/models/xai_test_model_1.pt` | Trained PyTorch Neural Network weights |
| `storage_data/artifacts/preprocessing/` | Preprocessed OULAD research splits (`splits.npz`) |
| `official_oulad_research_cohort.csv` | Sample cohort dataset for live CSV batch upload testing |

---

## 2. How to Run (Choose Your Preferred Method)

### Method A: One-Click Run (Windows — Recommended for Presentations)
1. Open the project folder in Windows File Explorer.
2. Double-click **`START_DEMO.bat`**.
3. The script will:
   - Confirm Python is installed.
   - Automatically install any missing packages (`fastapi`, `uvicorn`, `torch`, `pandas`, `numpy`, `python-multipart`).
   - Launch the PyTorch AI engine.
   - Automatically open your default web browser to:
     ```
     http://127.0.0.1:8000
     ```
4. **Keep the command prompt window open** during your presentation.

---

### Method B: Terminal / Command Line (Windows, macOS, or Linux)
If running from PowerShell, Command Prompt, or a macOS / Linux terminal:

1. Open your terminal in the project directory:
   ```bash
   cd "Dropout Prediction"
   ```
2. Install the minimal dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the server:
   ```bash
   python backend_lite.py
   ```
   *(Or on macOS/Linux: `bash start_demo.sh`)*
4. Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```

---

## 3. Walkthrough: What to Demonstrate to Your Professor

Once the dashboard loads at `http://127.0.0.1:8000`, guide your professor through the 5 core modules:

```
[ Navbar ] ──> 1. Topology ──> 2. Training Engine ──> 3. Batch Uploader ──> 4. Sandbox Predictor ──> 5. Benchmarks
```

### Module 1: Institutional Topology (Decentralized Federated Network)
* **What it shows**: Three simulated autonomous campus nodes:
  - **Node 1: Faculty of Engineering & Technology** (High VLE engagement)
  - **Node 2: Faculty of Social Sciences & Humanities** (Continuous coursework)
  - **Node 3: School of Business & Management** (Variable credit loads)
* **Key concept to explain**: Each institution's student records remain strictly local and private. No raw data ever leaves the college node.

### Module 2: Federated Training Engine (FedProx vs. FedAvg)
* **How to test**:
  1. Click **FedProx** (recommended) or **FedAvg**.
  2. Adjust **Aggregation Rounds** (e.g., 5 rounds) and **Proximal Regularization $\mu$** (e.g., 0.01).
  3. Click **"Launch Federated Training"**.
* **What it shows**:
  - Real-time client round synchronization and weight aggregation.
  - Convergence loss/accuracy progression charts.
  - **99.85% bandwidth reduction** compared to centralized training (only 4.69 KB model weights transferred per round).

### Module 3: Batch Cohort Risk Evaluator (Upload CSV)
* **How to test**:
  1. Scroll to the **"Batch Cohort Risk Evaluator"** section.
  2. Drag and drop the included file: `official_oulad_research_cohort.csv` (or click to browse).
* **What it shows**:
  - Instant vectorized PyTorch inference (< 50ms for the entire cohort).
  - Risk categorization: **High Risk (>=70%)**, **Medium Risk (40-69%)**, **Low Risk (<40%)**.
  - Interactive table showing predicted attrition probability, key risk drivers, and automated personalized intervention recommendations for each student.

### Module 4: Single-Student Interactive Sandbox & Explainable AI (XAI)
* **How to test**:
  1. Use the interactive sliders:
     - Lower **VLE Portal Clicks** to `< 200` and **Assessment Score** to `< 40` $\rightarrow$ Watch the probability spike to **High Risk**.
     - Increase **VLE Portal Clicks** to `800+` and **Assessment Score** to `85+` $\rightarrow$ Probability drops to **Low Risk (<15%)**.
* **What it shows**:
  - **SHAP-style Feature Attributions**: Visualizes which exact metrics contributed to the risk score (e.g., low portal logins, assessment failures, credit overload).
  - **Personalized Academic Learning Plan**: Generates actionable remediation steps, micro-learning pacing strategies, and milestone goals.

### Module 5: Empirical Benchmark Table
* **What it shows**:
  - A comparative scientific evaluation between **Centralized Baseline**, **FedAvg**, and **FedProx (Drift Mitigation)**.
  - Highlights how FedProx mitigates Non-IID client drift while preserving 100% data privacy.

---

## 4. Troubleshooting & FAQ

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| `'python' is not recognized as an internal or external command` | Python is not added to system environment variables | Reinstall Python and check the box **"Add Python to PATH"** on the first installer screen. |
| `Port 8000 already in use` | Another instance or server is running on port 8000 | Close any existing Python terminal windows, or open `backend_lite.py` and change `port=8000` to `port=8080`. |
| Browser shows *Connection Refused* initially | PyTorch is initializing weights in memory (~2-3s) | Wait 3 seconds and refresh the browser page (`F5`). |
| How to stop the server? | Server is actively serving requests | Press `Ctrl + C` in the terminal window, or simply close the command prompt window. |

---

## 5. Packaging & Submitting (Creating the ZIP File)

When sharing this project via USB, Google Drive, or email to your professor:

### What to INCLUDE in the ZIP:
- `START_DEMO.bat` and `start_demo.sh`
- `backend_lite.py`
- `requirements.txt`
- `frontend/dist/` (Crucial: contains the pre-compiled interactive web dashboard)
- `storage_data/` (Crucial: contains the trained PyTorch neural network weights and preprocessed OULAD splits)
- `official_oulad_research_cohort.csv` (Sample dataset for testing batch CSV uploads)
- `HOW_TO_RUN_GUIDE.md` and `README.md`

### What to EXCLUDE from the ZIP (Reduces file size from 5+ GB to ~25 MB!):
- `.venv`, `.venv_win`, `venv` (Do NOT include virtual environments — they contain machine-specific binary files and are gigabytes in size. The bat script will auto-install packages on any new machine).
- `frontend/node_modules/` (Not needed because `frontend/dist/` is already compiled).
- `.git/` (Optional version control history).

