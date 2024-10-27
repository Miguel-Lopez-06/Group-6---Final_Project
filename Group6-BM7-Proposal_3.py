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

st.header('------------------------------------------------------------')
st.header('Pie Chart of Outcome')

def pie_chart_Outcome():

  plt.pie(Outcome_counts_list, labels=Outcome_list, autopct='%1.1f%%')
  plt.title('Pie Chart of Outcome')
st.pyplot(plt)
# Clears the current figure
plt.clf()
pie_chart_Outcome()