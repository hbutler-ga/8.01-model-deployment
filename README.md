# Module 15 — Model Persistence and Streamlit Apps

**Session Time:** 120 minutes  
**Module Focus:** Train a simple model, save it with `joblib`, and use it inside a lightweight Streamlit app.

---

## Prerequisites

Before this lesson, students should be comfortable with:

- Loading data with pandas
- Selecting features and target variables
- Splitting data into training and testing sets
- Training a basic scikit-learn model
- Evaluating a regression model with metrics such as MAE or RMSE
- Running Python files from the terminal

---

## Learning Objectives

By the end of this lesson, students will be able to:

- Explain why trained models need to be saved before they can be reused
- Train a simple scikit-learn regression model
- Save a trained model using `joblib`
- Load a saved model in a separate Python file
- Build a simple Streamlit app that accepts user input
- Use a saved model to generate predictions inside a Streamlit app
- Explain why feature order and preprocessing consistency matter when using a saved model

---

## Lesson Overview

In the previous module, we trained and evaluated machine learning models in a notebook.

That is useful for development, but a notebook is not usually how other people interact with a model.

In this lesson, we will move one step closer to deployment by creating a small app around a trained model.

The workflow is:

```text
Train model → Save model → Load model → Build app → Generate predictions
```

This is not full production deployment. Instead, this is a beginner-friendly introduction to making a model reusable and interactive.

---

## Suggested Repository Structure

```text
module-15-model-persistence-streamlit/
│
├── data/
│   └── housing.csv
│
├── models/
│   └── housing_model.joblib
│
├── notebooks/
│   └── 01_train_model.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Session Breakdown

| Segment | Topic | Time |
|---:|---|---:|
| 1 | What does it mean to use a model outside a notebook? | 10 min |
| 2 | Train a simple scikit-learn model | 25 min |
| 3 | Save and load the model with `joblib` | 25 min |
| 4 | Build a simple Streamlit app | 40 min |
| 5 | Run and test the app locally | 10 min |
| 6 | Wrap-up and lab instructions | 10 min |

---

## Part 1: Why Save a Model?

When we train a machine learning model, the model learns patterns from the training data.

If we do not save the trained model, then we would need to retrain it every time we want to make a prediction.

That is not practical.

Saving a model lets us:

- Reuse it later
- Share it with another script or app
- Avoid retraining every time
- Keep the same trained version of the model
- Build tools around it, such as dashboards or prediction apps

In this lesson, we will use `joblib` to save and load our model.

---

## Part 2: Train a Simple Model

For this lesson, we will use a small housing price example.

The goal is to predict a home price using a few simple features.

### Example features

```text
sqft
bedrooms
bathrooms
age
```

### Example target

```text
price
```

### Training Code

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load the data
df = pd.read_csv("data/housing.csv")

# Select features and target
feature_cols = ["sqft", "bedrooms", "bathrooms", "age"]
X = df[feature_cols]
y = df["price"]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Generate predictions
preds = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print(f"MAE: ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
```

---

## Part 3: Save the Model with `joblib`

Once the model is trained, we can save it to the `models/` folder.

```python
import joblib

joblib.dump(model, "models/housing_model.joblib")

print("Model saved to models/housing_model.joblib")
```

This creates a file that stores the trained model.

That file can now be loaded in another notebook, Python script, or Streamlit app.

---

## Part 4: Save the Feature Names Too

A model expects the same features it was trained on.

This matters because the model does not understand our business problem. It only understands the numeric columns it received during training.

A safer approach is to save a small model package that includes both:

- The trained model
- The feature names used during training

```python
model_package = {
    "model": model,
    "feature_names": feature_cols,
    "metrics": {
        "mae": mae,
        "rmse": rmse
    }
}

joblib.dump(model_package, "models/housing_model_package.joblib")
```

Now we can load the model and the expected feature names together.

---

## Part 5: Load the Saved Model

In a new Python session, we can load the saved model package.

```python
import joblib
import pandas as pd

model_package = joblib.load("models/housing_model_package.joblib")

model = model_package["model"]
feature_names = model_package["feature_names"]

new_house = pd.DataFrame({
    "sqft": [1800],
    "bedrooms": [3],
    "bathrooms": [2],
    "age": [15]
})

prediction = model.predict(new_house[feature_names])

print(f"Predicted Price: ${prediction[0]:,.2f}")
```

The important idea is that the new data must contain the same feature columns used during training.

---

## Part 6: What is Streamlit?

Streamlit is a Python library for building lightweight web apps.

It is useful for data science projects because it allows us to quickly turn Python code into an interactive app.

In this lesson, we will use Streamlit to create a simple housing price prediction app.

The app will:

1. Load the saved model
2. Ask the user for housing information
3. Convert the user input into a pandas DataFrame
4. Generate a prediction
5. Display the predicted price

---

## Part 7: Build the Streamlit App

Create a file named `app.py`.

```python
import streamlit as st
import pandas as pd
import joblib

# Load the saved model package
model_package = joblib.load("models/housing_model_package.joblib")

model = model_package["model"]
feature_names = model_package["feature_names"]
metrics = model_package["metrics"]

# App title
st.title("Housing Price Predictor")

st.write(
    "Enter the details of a house below and the model will estimate its price."
)

# User inputs
sqft = st.number_input("Square Feet", min_value=500, max_value=6000, value=1800)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
age = st.number_input("Age of Home", min_value=0, max_value=150, value=15)

# Create input DataFrame
input_data = pd.DataFrame({
    "sqft": [sqft],
    "bedrooms": [bedrooms],
    "bathrooms": [bathrooms],
    "age": [age]
})

# Make sure columns are in the same order used during training
input_data = input_data[feature_names]

# Prediction button
if st.button("Predict Price"):
    prediction = model.predict(input_data)[0]
    st.subheader(f"Predicted Price: ${prediction:,.2f}")

# Optional model information
with st.expander("Model Information"):
    st.write("Model type: Linear Regression")
    st.write(f"MAE: ${metrics['mae']:,.2f}")
    st.write(f"RMSE: ${metrics['rmse']:,.2f}")
    st.write("Features used by the model:")
    st.write(feature_names)
```

---

## Part 8: Run the Streamlit App

From the terminal, run:

```bash
streamlit run app.py
```

Streamlit will start a local web app and provide a local URL.

Usually it looks something like:

```text
http://localhost:8501
```

Open the URL in your browser and test the app.

---

## Requirements File

Create a `requirements.txt` file:

```text
pandas
numpy
scikit-learn
joblib
streamlit
```

Install the requirements with:

```bash
pip install -r requirements.txt
```

---

## Important Teaching Point: Feature Consistency

A saved model is not magic.

It expects the same kind of input it saw during training.

That means:

- Same feature names
- Same feature order
- Same data types
- Same preprocessing steps

For example, if the model was trained with these columns:

```python
["sqft", "bedrooms", "bathrooms", "age"]
```

Then the app should pass those same columns into the model.

This is why we saved `feature_names` with the model package.

---

## Optional Extension: Add Input Validation

Students can improve the app by checking whether the inputs are realistic.

Example:

```python
if sqft < 500:
    st.warning("Square footage seems unusually low.")

if bedrooms > 8:
    st.warning("Bedroom count seems unusually high.")
```

This connects the lesson back to data validation and user input quality.

---

## Lab: Build Your Own Prediction App

### Goal

Create a simple Streamlit app that uses a saved machine learning model to make predictions.

### Instructions

1. Train a simple scikit-learn model in a notebook.
2. Evaluate the model using at least one metric.
3. Save the model using `joblib`.
4. Save the feature names with the model.
5. Create an `app.py` file.
6. Load the saved model inside the app.
7. Create Streamlit input widgets for each feature.
8. Generate and display a prediction.
9. Run the app locally.
10. Test the app with at least three different input examples.

---

## Lab Deliverables

Students should submit:

```text
notebooks/01_train_model.ipynb
app.py
models/housing_model_package.joblib
requirements.txt
README.md
```

The project README should include:

- What the app predicts
- What features the model uses
- How to install dependencies
- How to run the Streamlit app
- A short note about model performance

---

## Reflection Questions

1. Why do we save a trained model instead of retraining it every time?
2. What could go wrong if the app sends the model columns in the wrong order?
3. Why is it useful to save feature names with the model?
4. How is a Streamlit app different from a notebook?
5. What would need to change before this app could be used in a real production environment?

---

## Wrap-Up

In this lesson, we moved beyond training a model in a notebook.

We trained a simple model, saved it, loaded it, and used it in a small interactive app.

This workflow is a practical first step toward model deployment:

```text
Model development → Model persistence → User-facing prediction app
```

Students are now ready to use this pattern in their capstone projects.
