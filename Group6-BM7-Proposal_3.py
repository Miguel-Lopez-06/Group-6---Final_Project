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

st.write('We can see from the pie chart that 65.1% denotes 1 the presence of diabetes and 34.9% denotes 0 the absence of diabetes.')

st.header('------------------------------------------------------------')
st.header('Decision Tree Classifier')


st.pyplot(plt)
plt.clf()
