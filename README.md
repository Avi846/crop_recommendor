# 🌱 Crop Recommendation System

A **Streamlit-based Machine Learning app** that recommends the best crop to grow based on **soil nutrients (N, P, K)**, **climate conditions (temperature, humidity, rainfall)**, and **soil pH**.  
The system provides **data exploration, preprocessing, feature engineering, class balancing, model training, prediction, and results analysis** — all in one interactive dashboard.

---

## 🚀 Features
- 📊 **Data Exploration**: Summary statistics, crop distribution charts, dataset insights  
- 🔍 **Preprocessing**: Handle missing values, duplicates, and outliers  
- ⚙️ **Feature Engineering**: Create meaningful ratios, interactions, and seasonal indicators  
- ⚖️ **Class Balancing**: Apply SMOTE to handle imbalanced crop classes  
- 🤖 **Model Training**: Train multiple models (Random Forest, XGBoost, Logistic Regression, KNN, Neural Network) with GridSearchCV  
- 🔮 **Crop Prediction**: Input soil & climate parameters to get recommended crops with confidence scores  
- 📈 **Results Analysis**: Compare model performance across Accuracy, Precision, Recall, and F1-score  

---

## 🗂️ Project Structure
crop_recommendor/
│
├── crop_recommendation_system.py # Main Streamlit application
├── Crop_recommendation.csv # Default dataset (if available)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Ignore unnecessary files

## 📦 Installation & Setup
Create & activate virtual environment
Install dependencies
Run the app

📊 Dataset

The project expects a dataset with the following columns:
N — Nitrogen content in soil
P — Phosphorus content in soil
K — Potassium content in soil
temperature — Temperature in °C
humidity — Relative humidity in %
ph — Soil pH value
rainfall — Rainfall in mm
label — Crop name
Default dataset: Crop_recommendation.csv (if present).
👉 If not included, please provide your own dataset in the same format.

🌾 Usage

Upload your dataset or use the default one
Navigate via the sidebar:
Data Exploration
Preprocessing
Feature Engineering
Class Balancing
Model Training
Crop Prediction
Results Analysis
Enter soil and climate parameters in the Crop Prediction tab to get recommendations.
Review model performance in the Results Analysis section.

📈 Example Output

Recommended Crop: Rice (Confidence: 92%)
Top 5 alternatives with probabilities displayed as progress bars
Model comparison charts (Accuracy, Precision, Recall, F1-Score)

🛠️ Tech Stack

Frontend: Streamlit
Data Handling: Pandas, NumPy
Visualization: Matplotlib, Seaborn
Machine Learning: Scikit-learn, XGBoost, imbalanced-learn
Model Interpretation: SHAP
Persistence: Joblib

📌 Future Improvements

🌍 Integrate real-time weather API data
☁️ Deploy app on Streamlit Cloud / Render / Railway
🧠 Add more ML models & hyperparameter tuning
🔎 Add SHAP-based explainability in prediction tab

📜 License

This project is licensed under the MIT License.
Feel free to use and modify for your own research or applications.

🤝 Contributing

Contributions are welcome!
Fork the repo
Create a feature branch (feature/new-feature)
Commit changes
Push and create a Pull Request

🙌 Acknowledgements

Dataset inspired by Crop Recommendation Dataset
Built with ❤️ using Python & Streamlit
