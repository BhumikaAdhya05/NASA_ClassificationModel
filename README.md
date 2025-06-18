# NASA_ClassificationModel
Random Forest classification model trained on NASA turbine engine dataset to detect early signs of engine failure. Includes preprocessing, feature engineering, hyperparameter tuning (GridSearchCV), evaluation (confusion matrix, F1 score), and model export.

# NASA Turbine Engine Failure Classification

A machine learning classification project that uses Random Forest to predict turbine engine failures based on NASA's CMAPSS dataset. The goal is to identify engine health status and anticipate failure using sensor readings and operational settings.

## 📊 Dataset
The CMAPSS dataset includes multivariate time-series data from aircraft engines, with features such as:
- Operational Settings (3 columns)
- Sensor Measurements (21 columns)
- Unit Number (Engine ID)
- Time in Cycles (for each engine)

## ⚙️ Project Workflow
1. **Data Preprocessing**
   - Dropped non-informative and correlated features
   - Label encoded engine conditions

2. **Feature Engineering**
   - Aggregated sensor readings over time
   - Added cycle-based statistics

3. **Modeling**
   - RandomForestClassifier pipeline
   - Hyperparameter tuning with `GridSearchCV`
   - Balanced class weights to handle imbalanced data

4. **Evaluation**
   - Confusion Matrix
   - Accuracy, Precision, Recall, F1-Score
   - Classification report

5. **Model Export**
   - Trained model serialized using `pickle` for deployment

## 🧠 Technologies Used
- Python (pandas, numpy)
- scikit-learn (RandomForest, GridSearchCV)
- matplotlib, seaborn (visualization)
- pickle (model saving)

## 📈 Example Output
- **F1 Score:** 0.91
- **Test Accuracy:** 92%
- **Best Parameters:** `max_depth=10`, `n_estimators=200`, etc.

## 📦 Model Usage
To use the model:
```python
import pickle
with open('nasa_rf_model.pkl', 'rb') as file:
    model = pickle.load(file)
prediction = model.predict(new_data)

