# 🩺 Lung Cancer Survival Prediction

This project predicts the survival of lung cancer patients based on diagnostic and demographic data using machine learning techniques. It uses a structured dataset of patient records, including medical history, lifestyle factors, and treatment details.

---

## 📌 Objective

> **Build a system that can predict the survival of a patient given details of the patient. Explore the data to understand the features and figure out an approach.**

This project includes:
- End-to-end data preprocessing
- Feature engineering (treatment duration, encoding)
- Model training and evaluation
- Saving the model
- Making sample predictions

---

## 📁 Dataset Overview

The dataset contains 890,000 patient records with the following columns:

| Column | Description |
|--------|-------------|
| `age`, `gender`, `country`, `cancer_stage` | Demographics |
| `diagnosis_date`, `end_treatment_date`     | Used to calculate treatment duration |
| `family_history`, `smoking_status`         | Lifestyle & medical history |
| `bmi`, `cholesterol_level`                 | Physical metrics |
| `hypertension`, `asthma`, `cirrhosis`, `other_cancer` | Conditions |
| `treatment_type`                           | Treatment administered |
| `survived`                                 | Binary target (0 = did not survive, 1 = survived) |

---

## 🧠 ML Pipeline

- **Model**: `RandomForestClassifier` with class balancing
- **Preprocessing**:
  - Binary encoding (`yes/no`, `male/female`)
  - One-hot encoding (`country`, `stage`, `treatment`, `smoking_status`)
  - Treatment duration (in days)
- **Train/test split**: 80/20 stratified

---

## 🧪 Sample Prediction

After training, the model can make a prediction like this:

```python
sample_patient = pd.DataFrame([X_test.iloc[0]])
prediction = model.predict(sample_patient)[0]
print("Survived ✅" if prediction == 1 else "Did not survive ❌")
```
## 🚀 How to Run
#### 1.Clone the repository:
```
git clone https://github.com/your-username/lung-cancer-survival.git
cd lung-cancer-survival
```
#### 2.Install dependencies:
```pip install -r requirements.txt```
#### 3.Run the project:
```python lung_survival_predictor.py```
#### You will see:

Accuracy

Classification report

Sample patient prediction

Model saved to ```models/lung_survival_model.pkl```

## ✅ Folder Structure

LUNG_CANCER_SURVIVAL/
├── data/
│   └── dataset_med.csv
├── models/
│   └── lung_survival_model.pkl
├── lung_survival_predictor.py
├── eda.ipynb
├── README.md
├── requirements.txt

## 📌 Notes
This project is for educational purposes only. The model was trained on synthetic or anonymized data and is not intended for real-world medical use.

## 🙋‍♀️ Author
Made by Shravani Bande

> ⚠️ Note: The original dataset (`dataset_med.csv`) and model file (`lung_survival_model.pkl`) were too large for GitHub. Please contact the author or use your own data to run the project.
