[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-prediction--krishnendu-dutta.streamlit.app/)
# Customer Churn Prediction 📊

This project is an interactive web application that predicts customer churn using a machine learning model. Users can input customer details through a simple web interface, and the application provides a real-time prediction of whether the customer is likely to churn or not. The app is built with Python and deployed using Streamlit.

---

## ✨ Features

* **Interactive Interface:** Allows users to input customer data using sliders and dropdowns.
* **Real-Time Prediction:** Instantly predicts the churn probability based on the provided inputs.
* **Machine Learning Model:** Utilizes a pre-trained XGBoost model to make accurate predictions.

---

## 🛠️ Technologies Used

* **Language:** Python
* **Data Handling:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Web Application:** Streamlit
* **Version Control:** Git & GitHub

---

## ⚙️ How to Run Locally

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/DuttaKrishnendu/Customer-Churn-Prediction.git](https://github.com/DuttaKrishnendu/Customer-Churn-Prediction.git)
    cd Customer-Churn-Prediction
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run "streamlit_app_churn _prediction.py"
    ```