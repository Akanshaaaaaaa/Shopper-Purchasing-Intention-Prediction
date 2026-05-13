# ===============================================
# 1️⃣ IMPORTING REQUIRED LIBRARIES
# ===============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay
)

# ===============================================
# 2️⃣ PAGE CONFIGURATION & HELPER FUNCTIONS
# ===============================================

# Set up the Streamlit page
st.set_page_config(
    page_title="Shopper Intention Prediction",
    page_icon="🛒",
    layout="wide"
)

# Use cache to avoid reloading data on every interaction
@st.cache_data
def load_data(filepath="online_shoppers_intention.csv"):
    """Loads and preprocesses the dataset."""
    df = pd.read_csv(filepath)
    
    # Handle missing values (if any)
    df = df.dropna()
    
    # Correcting data types that are often read incorrectly
    df['Weekend'] = df['Weekend'].astype(bool)
    
    # Encode the target variable
    le = LabelEncoder()
    df['Revenue'] = le.fit_transform(df['Revenue'])
    
    return df

@st.cache_data
def get_preprocessor(df):
    """Creates a ColumnTransformer for preprocessing features."""
    
    # Identify numeric and categorical features (excluding target 'Revenue')
   
    numeric_features = df.select_dtypes(include=np.number).columns
    categorical_features = df.select_dtypes(include=['object', 'bool']).columns
    
    # Create preprocessing pipelines for both types
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    return preprocessor

def plot_metrics(y_test, y_pred, y_proba, model_name):
    """Plots Confusion Matrix, ROC Curve, and PR Curve."""
    
    col1, col2 = st.columns(2)
    
    # Confusion Matrix
    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', ax=ax)
        ax.set_title(f"{model_name} - Confusion Matrix")
        st.pyplot(fig)
        
    # ROC Curve
    with col2:
        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
        ax.set_title(f"{model_name} - ROC Curve")
        st.pyplot(fig)

    # Precision-Recall Curve
    with col1:
        st.subheader("Precision-Recall Curve")
        fig, ax = plt.subplots()
        PrecisionRecallDisplay.from_predictions(y_test, y_proba, ax=ax)
        ax.set_title(f"{model_name} - PR Curve")
        st.pyplot(fig)

# ===============================================
# 3️⃣ MAIN APPLICATION INTERFACE
# ===============================================

st.title("🛒 Online Shopper Purchasing Intention Prediction")
st.write(
    """
    **Project by Akansha (20421) for CSN344 - Machine Learning**
    
    This app predicts whether an online shopper will make a purchase ('Revenue')
    based on their browsing behavior. Explore the data and test different
    machine learning models using the options in the sidebar.
    """
)

# --- Sidebar for User Inputs ---
with st.sidebar:
    st.header("⚙️ Model Configuration")
    
    test_size = st.slider(
        "Test Set Size", 
        min_value=0.1, 
        max_value=0.5, 
        value=0.2, 
        step=0.05
    )
    
    model_choice = st.selectbox(
        "Select Model",
        ("Decision Tree", "Logistic Regression", "Random Forest")
    )
    
    use_smote = st.checkbox(
        "Use SMOTE (for Imbalanced Data)", 
        value=True
    )
    
    run_training = st.button("Train and Evaluate Model")

# --- Main Page Content ---

# Load and display data summary
st.header("1. Load and Inspect Dataset")
df = load_data()
st.dataframe(df.head())
st.info(f"Dataset Loaded: **{df.shape[0]} rows** and **{df.shape[1]} columns**.")

# 3. Exploratory Data Analysis (EDA)
st.header("2. Exploratory Data Analysis (EDA)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Revenue Distribution (Target)")
    fig, ax = plt.subplots()
    sns.countplot(x='Revenue', data=df, ax=ax, palette='viridis')
    ax.set_title("Class Distribution (0 = No Purchase, 1 = Purchase)")
    st.pyplot(fig)
    st.markdown(
        """
        **Observation:** The dataset is **highly imbalanced**.
        Significantly more sessions do not end in a purchase.
        This is why using SMOTE (in the sidebar) is recommended.
        """
    )
    
with col2:
    st.subheader("PageValues vs. Revenue")
    fig, ax = plt.subplots()
    sns.boxplot(x='Revenue', y='PageValues', data=df, ax=ax, palette='mako')
    ax.set_title("PageValues Distribution by Revenue")
    st.pyplot(fig)
    st.markdown(
        """
        **Observation:** 'PageValues' (average value of pages visited)
        is a very strong indicator of a purchase.
        """
    )
    
st.subheader("Correlation Heatmap")
fig, ax = plt.subplots(figsize=(12, 8))
# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=False, ax=ax)
st.pyplot(fig)

# 4. Model Training and Evaluation
st.header("3. Model Training and Evaluation")

if run_training:
    # --- Data Preparation ---
    st.subheader(f"Training: {model_choice}")
    
    # Define features (X) and target (y)
    X = df.drop('Revenue', axis=1)
    y = df['Revenue']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # --- Model Selection ---
    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    else: # Random Forest
        model = RandomForestClassifier(random_state=42)
        
    # --- Build Pipeline ---
    preprocessor = get_preprocessor(X)
    
    pipeline_steps = [
        ('preprocessor', preprocessor),
        ('model', model)
    ]
    
    # Add SMOTE to pipeline if selected
    if use_smote:
        pipeline_steps.insert(1, ('smote', SMOTE(random_state=42)))
        pipeline = ImbPipeline(steps=pipeline_steps)
        st.write("Using **SMOTE** to balance training data.")
    else:
        pipeline = Pipeline(steps=pipeline_steps)
        st.write("Training on **original (imbalanced)** data.")

    # --- Train Model ---
    with st.spinner("Training model... Please wait."):
        pipeline.fit(X_train, y_train)
    
    st.success(f"✅ Model training complete!")
    
    # --- Evaluation ---
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] # Probability for the positive class
    
    # Display metrics in columns
    st.subheader("Model Performance Metrics")
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Accuracy", f"{acc:.4f}")
    mcol2.metric("Precision", f"{prec:.4f}")
    mcol3.metric("Recall", f"{rec:.4f}")
    mcol1.metric("F1-Score", f"{f1:.4f}")
    mcol2.metric("ROC-AUC Score", f"{auc:.4f}")

    st.subheader("Classification Report")
    st.code(classification_report(y_test, y_pred))
    
    # --- Plot Metrics ---
    plot_metrics(y_test, y_pred, y_proba, model_choice)
    
    # --- Feature Importance (for tree-based models) ---
    if model_choice in ["Decision Tree", "Random Forest"]:
        st.subheader("Top 10 Feature Importances")
        try:
            # Get feature names from the preprocessor
            feature_names = preprocessor.get_feature_names_out()
            
            # Get importances from the model in the pipeline
            importances = pipeline.named_steps['model'].feature_importances_
            
            feat_imp = pd.Series(importances, index=feature_names).nlargest(10)
            
            fig, ax = plt.subplots()
            feat_imp.plot(kind='barh', ax=ax, color='teal')
            ax.set_title(f"Top 10 Features - {model_choice}")
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Could not retrieve feature importances. Error: {e}")

else:
    st.info("Click the 'Train and Evaluate Model' button in the sidebar to start.")
