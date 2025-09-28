import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import shap
import joblib
import warnings
import io
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #228B22;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">🌱 Comprehensive Crop Recommendation System</h1>', unsafe_allow_html=True)
    
    # File upload section at the top
    st.subheader("📁 Upload Your Dataset")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload your crop recommendation CSV file", type="csv", 
                                        help="File should contain columns: N, P, K, temperature, humidity, ph, rainfall, label")
    
    with col2:
        st.markdown("""
        **Expected Format:**
        - N, P, K (nutrients)
        - temperature, humidity, ph, rainfall
        - label (crop name)
        """)
    
    # Load data
    @st.cache_data
    def load_data(uploaded_file=None):
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ File uploaded successfully! Loaded {len(df)} records")
                return df
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
                return None
        else:
            try:
                df = pd.read_csv('Crop_recommendation.csv')
                st.info("📁 Using default dataset: Crop_recommendation.csv")
                return df
            except FileNotFoundError:
                st.warning("⚠️ No file uploaded and default file not found")
                return None
    
    df = load_data(uploaded_file)
    
    if df is None:
        return
    
    # Check if required columns exist
    required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing required columns: {missing_columns}")
        st.info("Please upload a file with the correct column structure.")
        return
    
    # Sidebar navigation (only show if data is loaded successfully)
    st.sidebar.title("Navigation")
    section = st.sidebar.radio("Go to:", [
        "📊 Data Exploration",
        "🔍 Data Preprocessing", 
        "⚙️ Feature Engineering",
        "⚖️ Class Balancing",
        "🤖 Model Training",
        "🔮 Crop Prediction",
        "📈 Results Analysis"
    ])
    
    # Display dataset info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Dataset Info")
    st.sidebar.write(f"**Records:** {len(df)}")
    st.sidebar.write(f"**Features:** {len(df.columns)}")
    st.sidebar.write(f"**Crops:** {df['label'].nunique()}")
    
    if section == "📊 Data Exploration":
        data_exploration(df)
    elif section == "🔍 Data Preprocessing":
        data_preprocessing(df)
    elif section == "⚙️ Feature Engineering":
        feature_engineering(df)
    elif section == "⚖️ Class Balancing":
        class_balancing(df)
    elif section == "🤖 Model Training":
        model_training(df)
    elif section == "🔮 Crop Prediction":
        crop_prediction()
    elif section == "📈 Results Analysis":
        results_analysis()

def data_exploration(df):
    st.markdown('<h2 class="section-header">📊 Dataset Exploration</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Overview")
        st.write(f"**Dataset Shape:** {df.shape}")
        st.write(f"**Number of unique crops:** {df['label'].nunique()}")
        st.write(f"**Features:** {list(df.columns)}")
        
        st.subheader("Crop Types")
        st.write(df['label'].unique())
    
    with col2:
        st.subheader("Basic Statistics")
        st.dataframe(df.describe())
    
    st.subheader("Dataset Info")
    
    # Fix for df.info() display
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
    
    # Target distribution
    st.subheader("Crop Distribution Analysis")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    target_counts = df['label'].value_counts()
    target_counts.plot(kind='bar', ax=ax1)
    ax1.set_title('Crop Distribution')
    ax1.tick_params(axis='x', rotation=45)
    
    target_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
    ax2.set_title('Crop Percentage Distribution')
    
    st.pyplot(fig)
    
    st.write("**Class Distribution:**")
    st.dataframe(target_counts)

def data_preprocessing(df):
    st.markdown('<h2 class="section-header">🔍 Data Preprocessing</h2>', unsafe_allow_html=True)
    
    # Check for missing values and duplicates
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Quality Check")
        st.write("**Missing values per column:**")
        missing_values = df.isnull().sum()
        st.dataframe(missing_values)
    
    with col2:
        st.write(f"**Duplicate rows:** {df.duplicated().sum()}")
        
        # Remove duplicates
        df_clean = df.drop_duplicates()
        st.write(f"**Dataset after removing duplicates:** {df_clean.shape}")
    
    # Outlier detection
    st.subheader("Outlier Analysis")
    
    def detect_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers
    
    outlier_results = []
    for column in df_clean.columns[:-1]:  # Exclude label
        outliers = detect_outliers(df_clean, column)
        outlier_results.append({
            'Feature': column,
            'Outliers': len(outliers),
            'Percentage': f"{len(outliers)/len(df_clean)*100:.2f}%"
        })
    
    st.dataframe(pd.DataFrame(outlier_results))
    
    # Store cleaned data in session state
    st.session_state.df_clean = df_clean
    return df_clean

def feature_engineering(df):
    st.markdown('<h2 class="section-header">⚙️ Feature Engineering</h2>', unsafe_allow_html=True)
    
    # Use cleaned data if available
    if 'df_clean' in st.session_state:
        df = st.session_state.df_clean
    
    # Separate features and target
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Encode target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    st.write(f"**Encoded classes:** {label_encoder.classes_}")
    
    # Create engineered features
    X_engineered = X.copy()
    
    # Nutrient ratios
    X_engineered['N_P_ratio'] = X['N'] / (X['P'] + 1e-8)
    X_engineered['N_K_ratio'] = X['N'] / (X['K'] + 1e-8)
    X_engineered['P_K_ratio'] = X['P'] / (X['K'] + 1e-8)
    
    # Climate-nutrient interactions
    X_engineered['temp_humidity_ratio'] = X['temperature'] / (X['humidity'] + 1e-8)
    X_engineered['nutrient_total'] = X['N'] + X['P'] + X['K']
    X_engineered['climate_stress'] = X['temperature'] * X['rainfall'] / (X['humidity'] + 1e-8)
    
    # Soil health indicators
    X_engineered['ph_nutrient_balance'] = X['ph'] * X_engineered['nutrient_total']
    X_engineered['rainfall_nutrient_ratio'] = X['rainfall'] / (X_engineered['nutrient_total'] + 1e-8)
    
    # Seasonal indicators
    def get_season(temp):
        if temp < 20: return 'cool'
        elif temp < 25: return 'moderate'
        else: return 'warm'
    
    X_engineered['season'] = X['temperature'].apply(get_season)
    season_encoded = pd.get_dummies(X_engineered['season'], prefix='season')
    X_engineered = pd.concat([X_engineered.drop('season', axis=1), season_encoded], axis=1)
    
    st.write(f"**Original features:** {list(X.columns)}")
    st.write(f"**Engineered features:** {list(X_engineered.columns)}")
    st.write(f"**Total features after engineering:** {X_engineered.shape[1]}")
    
    # Correlation analysis
    correlation_with_target = pd.DataFrame({
        'feature': X_engineered.columns,
        'correlation': [np.corrcoef(X_engineered[col], y_encoded)[0, 1] for col in X_engineered.columns]
    }).sort_values('correlation', key=abs, ascending=False)
    
    st.subheader("Top Correlated Features with Target")
    st.dataframe(correlation_with_target.head(10))
    
    # Store in session state
    st.session_state.X_engineered = X_engineered
    st.session_state.y_encoded = y_encoded
    st.session_state.label_encoder = label_encoder
    
    return X_engineered, y_encoded, label_encoder

def class_balancing(df):
    st.markdown('<h2 class="section-header">⚖️ Class Balancing</h2>', unsafe_allow_html=True)
    
    # Check if feature engineering is done
    if 'X_engineered' not in st.session_state:
        st.warning("⚠️ Please run Feature Engineering first!")
        return
    
    X_engineered = st.session_state.X_engineered
    y_encoded = st.session_state.y_encoded
    label_encoder = st.session_state.label_encoder
    
    # Class distribution
    class_distribution = pd.Series(y_encoded).value_counts().sort_index()
    class_names = label_encoder.classes_
    
    st.subheader("Class Distribution")
    dist_data = []
    for i, count in class_distribution.items():
        dist_data.append({'Crop': class_names[i], 'Samples': count})
    
    st.dataframe(pd.DataFrame(dist_data))
    
    # Apply SMOTE
    if st.button("Apply SMOTE for Class Balancing"):
        with st.spinner("Applying SMOTE..."):
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X_engineered, y_encoded)
            
            st.write(f"**Before SMOTE:** {X_engineered.shape[0]} samples")
            st.write(f"**After SMOTE:** {X_resampled.shape[0]} samples")
            
            # Show new distribution
            resampled_distribution = pd.Series(y_resampled).value_counts().sort_index()
            resampled_data = []
            for i, count in resampled_distribution.items():
                resampled_data.append({'Crop': class_names[i], 'Samples': count})
            
            st.dataframe(pd.DataFrame(resampled_data))
            
            # Store in session state for later use
            st.session_state.X_resampled = X_resampled
            st.session_state.y_resampled = y_resampled
            
            st.success("✅ SMOTE applied successfully!")

def model_training(df):
    st.markdown('<h2 class="section-header">🤖 Model Training</h2>', unsafe_allow_html=True)
    
    if 'X_resampled' not in st.session_state:
        st.warning("⚠️ Please run Class Balancing first!")
        return
    
    X_resampled = st.session_state.X_resampled
    y_resampled = st.session_state.y_resampled
    label_encoder = st.session_state.label_encoder
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X_resampled.columns)
    
    # Split data
    test_size = st.slider("Test Size", 0.1, 0.3, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df, y_resampled, test_size=test_size, random_state=42, stratify=y_resampled
    )
    
    st.write(f"**Training set:** {X_train.shape}")
    st.write(f"**Testing set:** {X_test.shape}")
    
    # Model selection
    st.subheader("Select Models for Training")
    
    models_to_train = st.multiselect(
        "Choose models:",
        ["Random Forest", "XGBoost", "Logistic Regression", "K-Nearest Neighbors", "Neural Network"],
        default=["Random Forest", "XGBoost"]
    )
    
    models_config = {
        'Random Forest': {
            'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
            'params': {'n_estimators': [100, 200], 'max_depth': [10, 20]}
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'params': {'n_estimators': [100, 200], 'max_depth': [3, 6]}
        },
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
            'params': {'C': [0.1, 1, 10]}
        },
        'K-Nearest Neighbors': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [3, 5, 7]}
        },
        'Neural Network': {
            'model': MLPClassifier(max_iter=1000, random_state=42),
            'params': {'hidden_layer_sizes': [(100,), (50,)]}
        }
    }
    
    if st.button("🚀 Train Models"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        best_models = {}
        model_results = []
        
        for i, model_name in enumerate(models_to_train):
            status_text.text(f"Training {model_name}...")
            progress_bar.progress((i) / len(models_to_train))
            
            config = models_config[model_name]
            
            # Grid search
            grid_search = GridSearchCV(
                config['model'], config['params'],
                cv=StratifiedKFold(n_splits=3),  # Reduced for speed
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            best_models[model_name] = grid_search.best_estimator_
            
            # Evaluate
            y_pred = best_models[model_name].predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            model_results.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Best Params': str(grid_search.best_params_)
            })
        
        progress_bar.progress(100)
        status_text.text("✅ Training completed!")
        
        # Display results
        results_df = pd.DataFrame(model_results)
        st.subheader("Model Performance")
        st.dataframe(results_df.sort_values('Accuracy', ascending=False))
        
        # Store in session state
        st.session_state.best_models = best_models
        st.session_state.results_df = results_df
        st.session_state.scaler = scaler
        st.session_state.X_scaled_df = X_scaled_df
        st.session_state.y_resampled = y_resampled
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        
        st.success("Models trained and ready for prediction!")

def crop_prediction():
    st.markdown('<h2 class="section-header">🔮 Crop Prediction</h2>', unsafe_allow_html=True)
    
    if 'best_models' not in st.session_state:
        st.warning("⚠️ Please train models first!")
        return
    
    st.subheader("Enter Soil and Climate Parameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        N = st.slider("Nitrogen (N) ppm", 0, 140, 50)
        P = st.slider("Phosphorus (P) ppm", 5, 145, 50)
        K = st.slider("Potassium (K) ppm", 5, 205, 50)
    
    with col2:
        temperature = st.slider("Temperature (°C)", 8.0, 43.0, 25.0)
        humidity = st.slider("Humidity (%)", 14.0, 100.0, 70.0)
        ph = st.slider("pH Level", 3.5, 9.9, 6.5)
    
    with col3:
        rainfall = st.slider("Rainfall (mm)", 20.0, 300.0, 100.0)
        model_choice = st.selectbox("Select Model", list(st.session_state.best_models.keys()))
    
    if st.button("🌾 Get Crop Recommendation"):
        with st.spinner("Analyzing parameters..."):
            # Prepare input
            input_dict = {
                'N': N, 'P': P, 'K': K, 'temperature': temperature,
                'humidity': humidity, 'ph': ph, 'rainfall': rainfall
            }
            
            # Get selected model
            model = st.session_state.best_models[model_choice]
            label_encoder = st.session_state.label_encoder
            scaler = st.session_state.scaler
            X_engineered = st.session_state.X_engineered
            
            # Preprocess input
            input_processed = preprocess_input(input_dict, X_engineered.columns)
            input_scaled = scaler.transform(input_processed)
            
            # Make prediction
            prediction_encoded = model.predict(input_scaled)[0]
            probabilities = model.predict_proba(input_scaled)[0]
            
            predicted_crop = label_encoder.classes_[prediction_encoded]
            confidence = probabilities[prediction_encoded]
            
            # Display results
            st.success(f"**Recommended Crop: {predicted_crop}** (Confidence: {confidence:.2%})")
            
            # Show top alternatives
            top_5_idx = np.argsort(probabilities)[-5:][::-1]
            st.subheader("Top 5 Recommendations:")
            
            for i, idx in enumerate(top_5_idx):
                crop_name = label_encoder.classes_[idx]
                crop_prob = probabilities[idx]
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    st.metric(label=f"{i+1}. {crop_name}", value=f"{crop_prob:.2%}")
                with col2:
                    progress_value = int(crop_prob * 100)
                    st.progress(progress_value)

def preprocess_input(input_dict, feature_names):
    """Preprocess input for prediction"""
    N = input_dict['N']
    P = input_dict['P']
    K = input_dict['K']
    temperature = input_dict['temperature']
    humidity = input_dict['humidity']
    ph = input_dict['ph']
    rainfall = input_dict['rainfall']
    
    # Create engineered features
    features = {
        'N': N, 'P': P, 'K': K, 'temperature': temperature,
        'humidity': humidity, 'ph': ph, 'rainfall': rainfall,
        'N_P_ratio': N / (P + 1e-8),
        'N_K_ratio': N / (K + 1e-8),
        'P_K_ratio': P / (K + 1e-8),
        'temp_humidity_ratio': temperature / (humidity + 1e-8),
        'nutrient_total': N + P + K,
        'climate_stress': temperature * rainfall / (humidity + 1e-8),
        'ph_nutrient_balance': ph * (N + P + K),
        'rainfall_nutrient_ratio': rainfall / (N + P + K + 1e-8)
    }
    
    # Add seasonal indicators
    season = 'cool' if temperature < 20 else 'moderate' if temperature < 25 else 'warm'
    for season_type in ['cool', 'moderate', 'warm']:
        features[f'season_{season_type}'] = 1 if season == season_type else 0
    
    # Ensure correct order
    feature_vector = [features[col] for col in feature_names]
    
    return np.array(feature_vector).reshape(1, -1)

def results_analysis():
    st.markdown('<h2 class="section-header">📈 Results Analysis</h2>', unsafe_allow_html=True)
    
    if 'results_df' not in st.session_state:
        st.warning("⚠️ Please train models first!")
        return
    
    results_df = st.session_state.results_df
    
    # Model comparison chart
    st.subheader("Model Performance Comparison")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x_pos = np.arange(len(metrics))
    
    for model_name in results_df['Model']:
        model_metrics = results_df[results_df['Model'] == model_name][metrics].values[0]
        ax.plot(metrics, model_metrics, marker='o', label=model_name, linewidth=2)
    
    ax.set_title('Model Performance Comparison')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Best model info
    best_model_name = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
    st.subheader(f"Best Performing Model: {best_model_name}")
    
    best_model_stats = results_df[results_df['Model'] == best_model_name].iloc[0]
    st.write(f"**Accuracy:** {best_model_stats['Accuracy']:.4f}")
    st.write(f"**Precision:** {best_model_stats['Precision']:.4f}")
    st.write(f"**Recall:** {best_model_stats['Recall']:.4f}")
    st.write(f"**F1-Score:** {best_model_stats['F1-Score']:.4f}")

if __name__ == "__main__":
    main()