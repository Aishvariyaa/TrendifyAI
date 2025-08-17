# 👗 TrendifyAI – AI-powered trend & sales forecasting

### 📌 Overview  
This project focuses on **predicting sales** for a **fashion brand** using **machine learning algorithms**! 🛍️📈  
By analyzing **historical sales data**, the goal is to forecast future sales trends and optimize business decisions.  

### 🔍 Key Steps  
✅ **Dataset Preprocessing:**  
   - Handling **missing values** using **imputation** techniques.  
   - Encoding **categorical variables** with **one-hot encoding**.  
✅ **Model Training:**  
   - **Linear Regression** 📊  
   - **Random Forest Regressor** 🌲  
   - **Gradient Boosting Regressor** 🔥  
✅ **Model Evaluation:**  
   - Used **Mean Squared Error (MSE)** to compare performance.  
✅ **Final Model Selection:**  
   - Chose the **best-performing model** for accurate sales predictions.  

### 📂 Project Structure  
```
Sales-Prediction-Fashion-Brand/
│── README.md  # Documentation  
│── ML model.py  # Model Training & Evaluation  
│── ProductSalesTestingData.csv  # Fashion Brand testing Data  
│── ProductSalesTestingData.csv  # Fashion Brand training Data   
```

### 📊 Dataset  
For this project, we use a **sales dataset** containing:  

📌 **Features**:  
- **Product Attributes:** Product ID, Name, Price, Style, Fabric, Brand, Fabric Type  
- **Sales History:** Total Sales, First Month Sale, Second Month Sale  
- **External Factors:** Season, Pattern, Wash Case, Color Group  

🎯 **Target Variable**: **Third Month Sale**  

### 🔧 Technologies Used  
🔹 Python  
🔹 Pandas & NumPy (Data Processing)  
🔹 Scikit-learn (ML Models & Evaluation)  
🔹 Matplotlib & Seaborn (Data Visualization)  

### 📜 How to Run the Project?  
#### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Aishvariyaa/TrendifyAI.git
cd TrendifyAI 
```  

#### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```  

#### 3️⃣ Download Dataset  
📌 Download the dataset from [Kaggle](#)  and place it in the project folder.  

#### 4️⃣ Run the Script  
```bash
python ML model.py
```  

### 📈 Model Performance  
📌 **Model Comparison (MSE Scores)**  
```
Linear Regression:  0.00  ✅ (Best Model)  
Random Forest Regressor:  1.57  
Gradient Boosting Regressor:  3.79  
```  

📌 **Key Observations**  
- **Linear Regression** performed the best with an **MSE of 0.00**, making it the most accurate model.  
- **Random Forest Regressor** followed with an MSE of **1.57**, showing moderate performance.  
- **Gradient Boosting Regressor** had the highest MSE (**3.79**), indicating a lower accuracy.  

### 📌 Next Steps  
🔹 Improve feature selection and engineering for better generalization.  
🔹 Tune **hyperparameters** to enhance performance.  
🔹 Incorporate **time-series forecasting techniques** for improved long-term sales predictions.  
