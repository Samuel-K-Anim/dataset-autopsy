🩺 Dataset Autopsy Lab (v1.0)

A modular, Object-Oriented Data Science tool designed to automate Exploratory Data Analysis (EDA) and Data Cleaning. Acting as both a "Coroner" and a "Surgeon," this application helps users diagnose data quality issues and interactively fix them without writing a single line of code.

🔗 **[Live Demo App](https://dataset-autopsy-lab-ask.streamlit.app)**
---

## 🚀 Key Features

### 1. 🔍 The Data Coroner (Diagnosis)

Upload any CSV or Excel file to get an instant health check:

* **Vitals Check:** Automatically detects missing values, duplicates, and memory usage.

* **Abnormality Detection:** Identifies statistical outliers (using IQR method) and highly skewed distributions.

* **Leakage Detection:** Scans for features that are suspiciously correlated (>95%) with your target variable, preventing model cheating.

* **ML Readiness Score:** A gamified score (0-100) that tells you if your data is ready for Machine Learning.

### 2. 🧪 The Treatment Room (Surgery)

An interactive interface to clean your data step-by-step:

* **Smart Imputation:** Fill missing values with Median, Mean, Mode, or a specific custom value.

* **Outlier Capping:** Apply Winsorization to cap extreme values at the 5th and 95th percentiles.

* **Skewness Correction:** Automatically apply Log Transformation (np.log1p) to normalize skewed features.

* **Categorical Encoding:** Convert text columns to numbers using One-Hot or Label Encoding strategies.

* **Date Conversion:** Detects and converts object columns to datetime format automatically.

### 3. 📝 Professional Reporting

* **PDF Export:** Generates a downloadable PDF "Autopsy Report" that documents every issue found and every cleaning action taken.

* **Change Log:** Tracks your edit history (e.g., "Renamed 3 columns", "Dropped 50 rows") for full reproducibility.

---

## 🛠️ Tech Stack

* **Architecture:** Modular Python (OOP) separation between Logic (autopsy_engine.py) and UI (app.py).

* **Frontend:** Streamlit

* **Data Processing:** Pandas, NumPy, Scikit-Learn

* **Statistics:** SciPy

* **Visualization:** Plotly Express

* **Reporting:** FPDF

---

## 💻 How to Run Locally

1. **Clone the repository**

```bash
    git clone [https://github.com/Samuel-K-Anim/dataset-autopsy.git](https://github.com/Samuel-K-Anim/dataset-autopsy.git)
    cd dataset-autopsy
```


2. **Install dependencies**

```bash
    pip install -r requirements.txt
```


3. **Run the App**

```bash
    streamlit run app.py
```


## 📁 Project Structure
```text
├── app.py                 # The Face: Handles UI, Session State, and User Interaction
├── autopsy_engine.py      # The Brain: Contains DataCoroner and DataHealer classes
├── report_generator.py    # The Reporter: Handles PDF generation logic
├── requirements.txt       # List of Python libraries
└── .streamlit/
    └── config.toml        # Server configuration (Upload limits)


🌟 Acknowledgements

Built as part of a Data Science portfolio project demonstrating advanced EDA, Object-Oriented Programming, and Software Engineering principles in Data Science.

Author: Samuel K. Anim