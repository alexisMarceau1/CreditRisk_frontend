# CreditRisk Frontend

This repository contains the **frontend application** for the **CreditRisk** project, built using [Streamlit](https://streamlit.io/). The dashboard provides interactive visualizations and interpretability for credit risk predictions.

---

## Features

- **Interactive Dashboard**:
  - Visualizes credit risk predictions.
  - Displays thresholds and detailed customer insights.

- **Customer Comparison**:
  - Compare customer data against others or a global dataset.
  
- **Interpretability**:
  - Provides SHAP-based explanations of predictions.

---

## How to Run the Frontend Locally

Follow these instructions to set up and run the frontend:

### Prerequisites

Before you begin, ensure the following:

1. **Python 3.8 or higher** is installed.
2. The **backend API** is running. You must set up and launch the backend before running the frontend.

### Step-by-Step Instructions

1. **Clone the Repository**:
   Clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/alexisMarceau1/CreditRisk_frontend.git
   cd CreditRisk_frontend
 2. **Install Dependencies**:
    Install the required Python packages using::
    ```bash
    pip install -r requirements.txt
 3. **Start the Backend API**:
    Ensure the backend is running at http://127.0.0.1:8000. To do this:
    - Clone and navigate to the CreditRisk Backend Repository.
    - Follow the setup instructions in that repository.
    - Run the backend using:
        ```bash
        uvicorn api:app --reload
 4. **Run the Frontend Dashboard**:
    Launch the Streamlit application by running:
    ```bash
    streamlit run dash.py
  5. **Access the Dashboard**:
    Open your browser and navigate to the following URL:
    ```bash
    [streamlit run dash.py](http://localhost:8501)
  
  

  

