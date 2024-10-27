import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import squarify
from wordcloud import WordCloud
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import plotly.graph_objects as go
from io import StringIO

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

# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define features and target (assuming the dataset is already loaded from the previous code)
X = df[features]
y = df['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Predict the outcomes on the test set
y_pred_tree = dt_classifier.predict(X_test)
tree_accuracy = accuracy_score(y_test, y_pred_tree)

# Print accuracy
print(f'Decision Tree Accuracy: {tree_accuracy * 100:.2f}%')

# Confusion matrix for Decision Tree
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_tree, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Decision Tree')
plt.show()

# Feature importance plot
feature_importance = dt_classifier.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importance for Decision Tree")
plt.show()



