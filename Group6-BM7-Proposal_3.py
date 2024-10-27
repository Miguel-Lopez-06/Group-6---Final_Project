from sklearn.base import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[features]
y = df['Outcome']

X

y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train.shape

X_train.head()

X_test.shape

X_test.head()

y_train.shape

y_train.head()

y_test.shape

y_train.head()

dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

y_pred_tree = dt_classifier.predict(X_test)
tree_accuracy = accuracy_score(y_test, y_pred_tree)

print(f'Accuracy: {tree_accuracy * 100:.2f}%')

feature_importance = dt_classifier.feature_importances_

feature_importance

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
})

# Sort the DataFrame by importance for better readability
importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Display the resulting DataFrame
print(importance_df)

plt.figure(figsize=(10, 6))
feature_importances = pd.Series(dt_classifier.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importance for Decision Tree")
st.pyplot(plt)
plt.clf()
