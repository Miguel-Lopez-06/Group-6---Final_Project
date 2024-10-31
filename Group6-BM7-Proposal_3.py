import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import altair as alt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import plotly.graph_objects as go
from io import StringIO

st.title('Group 6 BM7 - Proposal #3')
st.markdown('`Diabetes Dataset`')
st.header('------------------------------------------------------------')

df = pd.read_csv("diabetes.csv")

df

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

st.write('We can see from the pie chart that 65.1% denotes 1 the presence of diabetes and 34.9% denotes 0 the absence of diabetes.')

st.header('------------------------------------------------------------')
st.header('Model Training ( Decision Tree Classifier )')

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

plt.figure(figsize=(10, 6))
feature_importances = pd.Series(dt_classifier.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importance for Decision Tree")
st.pyplot(plt)
plt.clf()

cm = confusion_matrix(y_test, y_pred_tree)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Decision Tree")
st.pyplot(plt)
plt.clf()

y_pred_proba_tree = dt_classifier.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_tree)
roc_auc_tree = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', label='AUC = %0.2f' % roc_auc_tree)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Decision Tree")
plt.legend(loc="lower right")
st.pyplot(plt)
plt.clf()