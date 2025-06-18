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

# 🚀 NASA Server Log Analysis

This project analyzes web server access logs from NASA using Python. It processes the dataset to derive insights such as the number of unique hosts, frequency of requests, error percentages, and data transfer statistics. Ideal for practicing data wrangling and log file analysis.

---

## 📊 Project Objectives

- Parse NASA HTTP server logs (July 1995 dataset)
- Extract meaningful insights:
  - Unique host count
  - Total requests
  - 404 error frequency and analysis
  - Daily request counts
  - Bandwidth consumption by day

---

## 🧰 Tools & Libraries Used

- Python 3
- Pandas
- Matplotlib / Seaborn (optional for visualization)
- Regular Expressions (`re`) for log parsing

---

## 📁 Dataset

- **Source**: NASA HTTP Logs July 1995  
- **Format**: Raw `.log` text file with entries like:  
  `199.72.81.55 - - [01/Jul/1995:00:00:01 -0400] "GET /history/apollo/ HTTP/1.0" 200 6245`

---

## 🧠 Key Operations

1. **Reading and Parsing the Log File**  
   Uses `pandas` and string operations or regex to split each log line into fields:
   - Host
   - Timestamp
   - Request method & URL
   - HTTP status code
   - Bytes transferred

2. **Data Cleaning**  
   - Handles missing or malformed lines
   - Converts numeric columns
   - Parses timestamps for time-based aggregation

3. **Data Analysis**  
   - Unique hosts count
   - Total number of 404 errors
   - Top 5 URLs causing 404 errors
   - Errors per day
   - Bytes served per day

4. **(Optional)**: Visualizations  
   Can include line plots or bar graphs showing trends over time.

---

## 📈 Sample Insights

- How many unique hosts visited NASA’s site in July 1995
- What percentage of total requests resulted in 404 errors
- On which day did the server transfer the most data
- Which pages most frequently failed (404)

---

## 💡 Use Cases

- Real-world log data analysis
- Data wrangling practice with unstructured text
- HTTP log format parsing using regex
- Error monitoring and bandwidth usage analysis

---

## 🏁 How to Run

1. Clone this repository or open in Google Colab / Jupyter
2. Place the dataset in the same directory as the notebook
3. Run each cell in sequence
4. Modify the analysis as needed

---

## 📜 License

Open-source, for educational and research use.

---

## 🙋‍♀️ Author

**Bhumika Adhya**  
Aspiring Data Scientist | Python & Data Analysis Enthusiast  
[GitHub](https://github.com) | [LinkedIn](https://linkedin.com)


## 📦 Model Usage
To use the model:
```python
import pickle
with open('rf_model.pkl', 'rb') as file:
    model = pickle.load(file)
prediction = model.predict(new_data)

