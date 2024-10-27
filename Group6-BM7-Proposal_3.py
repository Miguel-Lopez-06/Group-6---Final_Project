import streamlit as st  # For creating interactive web apps
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.tree import DecisionTreeClassifier  # Decision Tree model
from sklearn.metrics import accuracy_score  # To calculate accuracy
import seaborn as sns  # For statistical data visualization
import plotly.graph_objects as go  # For interactive visualizations with Plotly
from io import StringIO  # For handling string data as file-like objects

# Optional imports, if you need these specific functionalities
import altair as alt  # For declarative statistical visualizations
import squarify  # For creating tree maps
from wordcloud import WordCloud  # For generating word clouds
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting with Matplotlib
import networkx as nx  # For network graph visualizations

st.title('Group 6 BM7 - Proposal #3')
st.markdown('`Diabetes Dataset`')
st.header('------------------------------------------------------------')

df = pd.read_csv("diabetes.csv")

df
st.write('This Bar Chart shows the types of CPU that Apple used in their laptops.')

st.header('------------------------------------------------------------')
st.header('Pie Chart of Outcome')

df.head()

df.info()

df['Outcome'].unique()

Outcome_counts = df['Outcome'].value_counts()
print(Outcome_counts)

Outcome_counts = df['Outcome'].value_counts()
Outcome_counts_list = Outcome_counts.tolist()
print(Outcome_counts_list)

Outcome_list = df['Outcome'].unique().tolist()
print(Outcome_list)

def pie_chart_Outcome():

    plt.pie(Outcome_counts_list, labels=Outcome_list, autopct='%1.1f%%')
    plt.title('Pie Chart of Outcome')
    st.pyplot(plt)
    plt.clf()

pie_chart_Outcome()

st.write('This Bar Chart shows the types of CPU that Apple used in their laptops.')

st.header('------------------------------------------------------------')
st.header('Decision Tree Classifier')

# Define features and target variable
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[features]
y = df['Outcome']

# Display feature and target data
st.write("Features (X):", X)
st.write("Target (y):", y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display shapes of train and test datasets
st.write("X_train shape:", X_train.shape)
st.write("X_train head:", X_train.head())
st.write("X_test shape:", X_test.shape)
st.write("X_test head:", X_test.head())
st.write("y_train shape:", y_train.shape)
st.write("y_train head:", y_train.head())
st.write("y_test shape:", y_test.shape)
st.write("y_test head:", y_test.head())

# Initialize and train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions and calculate accuracy
y_pred_tree = dt_classifier.predict(X_test)
tree_accuracy = accuracy_score(y_test, y_pred_tree)

# Display the accuracy
st.write(f'Accuracy: {tree_accuracy * 100:.2f}%')

# Calculate and display feature importance
feature_importance = dt_classifier.feature_importances_

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

st.write("Feature Importances:", importance_df)

# Plot the feature importance
plt.figure(figsize=(10, 6))
feature_importances = pd.Series(dt_classifier.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importance for Decision Tree")
st.pyplot(plt)
plt.clf()




