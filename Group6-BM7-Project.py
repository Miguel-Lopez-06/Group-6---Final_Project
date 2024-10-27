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

st.title('Data Visualization Using Streamlit')
st.markdown('`Group 6 BM7`')
st.header('------------------------------------------------------------')


df = pd.read_csv("laptop_price-dataset.csv")

df

st.header('------------------------------------------------------------')
df.info

st.header('------------------------------------------------------------')

#Graph 1
st.header('Bar Chart of Most Common CPU Types Used by Apple')

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'laptop_price-dataset.csv'  # Replace with your file path in Google Colab
laptop_data = pd.read_csv("laptop_price-dataset.csv")

# Assuming your data is in a pandas DataFrame named 'df'
# If your DataFrame has a different name, replace 'df' with the actual name
apple_laptops = df[df['Company'] == 'Apple']
apple_cpu_count = apple_laptops['CPU_Type'].value_counts()

# Plot
plt.figure(figsize=(10, 6))
apple_cpu_count.plot(kind='bar', color='skyblue')
plt.title('Bar Chart of Most Common CPU Types Used by Apple')
plt.xlabel('CPU Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
st.pyplot(plt)
# Clears the current figure
plt.clf()

st.write('This Bar Chart shows the types of CPU that Apple used in their laptops.')
st.write('**Observation**: Intel Core i5: Because of the balance between the performance and efficiency of this CPU, it is the most used CPU of Apple in their MacBooks. Intel Core i7: This CPU appears more on the higher-end models of Macbook like the Macbook Pro.')
st.header('------------------------------------------------------------')

#Graph2
st.header('Violin Plot of Weight Distribution of Laptops')

import seaborn as sns
import matplotlib.pyplot as plt

# 3. Violin Plot for Weight Distribution of Laptops
plt.figure(figsize=(10, 6))  # Set the size of the plot

# Assuming your DataFrame is named 'df'
sns.violinplot(data=df, x='Weight (kg)', color='pink')  # Create a violin plot for laptop weights

plt.title('Violin Plot of Weight Distribution of Laptops')  # Set the title of the plot
plt.xlabel('Weight (kg)')  # Label the x-axis
plt.tight_layout()  # Adjust layout for better visualization
st.pyplot(plt)
# Clears the current figure
plt.clf()

st.write('The weight distribution of laptops is displayed in this violin plot, emphasizing the areas where the bulk of computers fall within a particular weight range.')
st.write('**Observation:** The majority of computers weigh between 1.0 and 2.5 kg, with bulkier models weighing between 1.8 kg and 1.3–1.5 kg at their highest points. Ultra-heavy laptops are rare, as only a small percentage of computers weigh more than 2.5 kg.')
st.header('------------------------------------------------------------')

#Graph3
st.header('Bubble Chart: RAM vs CPU Frequency vs Price')

import matplotlib.pyplot as plt

# Bubble Chart: RAM vs CPU Frequency vs Price
plt.figure(figsize=(10, 6))
plt.scatter(df['CPU_Frequency (GHz)'], df['RAM (GB)'],
            s=df['Price (Euro)']/10, alpha=0.6,
            c=df['Price (Euro)'], cmap='coolwarm', marker='o')

plt.title('Bubble Chart: RAM vs CPU Frequency vs Price', fontsize=14)
plt.xlabel('CPU Frequency (GHz)', fontsize=12)
plt.ylabel('RAM (GB)', fontsize=12)
plt.colorbar(label='Price (Euro)')
plt.grid(True)
st.pyplot(plt)
# Clears the current figure
plt.clf()

st.write('This bubble chart reveals a clear correlation between price and performance (RAM and CPU frequency) can be seen in this bubble chart. Since premium components are more expensive, high-end laptops are priced much higher, while budget versions with lower performance are less expensive.')
st.write('**Observation:** Generally speaking, laptops with CPU frequencies higher than 3 GHz and more RAM than 16 GB cost more. These are probably meant for customers who need a lot of processing power, such developers, gamers, or people who work on graphics-intensive projects.')
st.header('------------------------------------------------------------')

#Graph4
st.header('Area Chart: Average Laptop Price by Company')

# Area Chart: Distribution of Prices across Companies
df_grouped = df.groupby('Company')['Price (Euro)'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
df_grouped.plot(kind='area', color='skyblue', alpha=0.6)
plt.title('Area Chart: Average Laptop Price by Company', fontsize=14)
plt.xlabel('Company', fontsize=12)
plt.ylabel('Average Price (Euro)', fontsize=12)
plt.grid(True)
st.pyplot(plt)
# Clears the current figure
plt.clf()

st.write('The area chart illustrates the varied price tactics used by different laptop manufacturers. As a luxury tech brand, Apple places a premium price point on its products, but other firms target a wider market with a variety of products that range from high-end laptops to low-cost models.')
st.write('**Observation:** When it comes to typical laptop pricing, Apple leads the competition by a wide margin. This is a reflection of Apple focus on the premium end of the market, where devices like the MacBook Pro and MacBook Air are renowned for their luxury construction, cutting-edge features, and devoted customer bases.')
st.header('------------------------------------------------------------')

#Graph5
st.header('3D Surface Plot: Screen Size, Weight, and Price')

from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 3D Surface Plot: Screen Size, Weight, and Price
fig = plt.figure(figsize=(10, 11))
ax = fig.add_subplot(111, projection='3d')

X = df['Inches']
Y = df['Weight (kg)']
Z = df['Price (Euro)']
ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_title('3D Surface Plot: Screen Size, Weight, and Price', fontsize=14)
ax.set_xlabel('Screen Size (Inches)', fontsize=10)
ax.set_ylabel('Weight (kg)', fontsize=10)
ax.set_zlabel('Price (Euro)', fontsize=10)
st.pyplot(plt)
# Clears the current figure
plt.clf()

st.write('The 3D surface plot shows that weight and screen size are reliable indicators of cost. Generally speaking, larger, heavier laptops have more powerful hardware, which drives up the price. On the other hand, portability is a common goal of lighter, smaller laptops, the cost of which varies based on the brand and internal parts.')
st.write('**Observation:** Higher-end laptops typically have larger screens (15+ inches) and weigh more (2+ kg). This implies that more expensive parts, such sharper screens, potent GPUs, and premium materials, are frequently found in larger laptops.')
st.header('------------------------------------------------------------')

#Graph6
st.header('Bar Chart of Distribution of Laptop Types')

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'laptop_price-dataset.csv'  # Replace with your file path in Google Colab
laptop_data = pd.read_csv("laptop_price-dataset.csv")

# Create a bar chart for the distribution of laptop types (TypeName)
plt.figure(figsize=(10,6))
laptop_data['TypeName'].value_counts().plot(kind='bar', color='orange', edgecolor='black')
plt.title('Bar Chart of Distribution of Laptop Types')
plt.xlabel('Laptop Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.grid(True)
st.pyplot(plt)
# Clears the current figure
plt.clf()

st.write('This is a Bar graph of the distribution of laptop types.')
st.write('**Observation:** In this graph, there are different laptop types namingly Notebook, Gaming, Ultrabook, Netbook, and other more. Notebook has a visible lead among all the laptop types in terms of distribution, as the distribution is around 700. Next is Gaming and Ultrabook laptops which have around 150-250 distribution. Workstation and Netbook had the two lowest distributions of laptops with less than 100. Notebook is one of the most famous types of laptops. Notebooks are lightweight and smaller which makes it portable and good for basic tasks like web browsing. Gaming laptops are designed to handle playing video games. It is also good for video editing because of the higher RAM and better GPU than other laptops.')
st.header('------------------------------------------------------------')

#Graph7
st.header('Scatter Plot of RAM vs Price')

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'laptop_price-dataset.csv'  # Replace with your file path in Google Colab
laptop_data = pd.read_csv("laptop_price-dataset.csv")

# Scatter plot: RAM vs Price
plt.figure(figsize=(10,6))
plt.scatter(laptop_data['RAM (GB)'], laptop_data['Price (Euro)'], color='green', alpha=0.6)
plt.title('Scatter Plot of RAM vs Price')
plt.xlabel('RAM (GB)')
plt.ylabel('Price (Euro)')
plt.grid(True)
st.pyplot(plt)
# Clears the current figure
plt.clf()

st.write('This is a scatter plot of RAM vs Price')
st.write('**Observation:** The graph shows a positive correlation between the Price and RAM as the RAM increases the price of the laptop also increases. 32 GB RAM laptops are the most expensive out of the other RAM choices. One outlier is the 64 GB RAM laptop costs around 4,000 euros which is slightly cheaper than other 32 GB RAM laptops. The price of the laptop also varies based on the different components installed or features.')
st.header('------------------------------------------------------------')

#Graph8
st.header('Line Chart of Average Laptop Price by Operating System')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st

# Convert Price (Euro) to numeric if necessary
df['Price (Euro)'] = pd.to_numeric(df['Price (Euro)'], errors='coerce')

# Drop rows with missing prices or operating system information
df = df.dropna(subset=['Price (Euro)', 'OpSys'])

# Group the data by operating system and calculate the average price
avg_price_per_os = df.groupby('OpSys')['Price (Euro)'].mean().reset_index()

# Sort by price for better visualization
avg_price_per_os = avg_price_per_os.sort_values(by='Price (Euro)')

# Plot the Line Chart for Average Laptop Price by Operating System
plt.figure(figsize=(10, 6))
sns.lineplot(x='OpSys', y='Price (Euro)', data=avg_price_per_os, marker='o', color='green')
plt.title('Line Chart of Average Laptop Price by Operating System')
plt.xlabel('Operating System')
plt.ylabel('Average Price (Euro)')
plt.grid(True)
plt.xticks(rotation=45)

# Display the plot using Streamlit
st.pyplot(plt)

# Clears the current figure
plt.clf()



st.write('This line chart compares the average price of laptops running different operating systems.')
st.write('**Observation:** Laptops that run on Android OS, Chrome OS, Linux, or have no operating system at all are significantly cheaper, while Windows laptops and MacBooks are more expensive. It is also notable that laptops that run on Windows 7 are more expensive than ones that run on Windows 10 and 10 S.')
st.header('------------------------------------------------------------')


#Graph9
st.header('Histogram of Distribution of Laptop Prices')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load the CSV file
df = pd.read_csv('laptop_price-dataset.csv')

# Convert Price (Euro) to numeric if necessary (in case it's read as string)
df['Price (Euro)'] = pd.to_numeric(df['Price (Euro)'], errors='coerce')

# Drop rows with missing prices (if any)
df = df.dropna(subset=['Price (Euro)'])

# Plot the Histogram for Distribution of Laptop Prices
plt.figure(figsize=(10, 6))
sns.histplot(df['Price (Euro)'], bins=30, kde=True, color='skyblue')
plt.title('Histogram of Distribution of Laptop Prices')
plt.xlabel('Price (Euro)')
plt.ylabel('Frequency')
plt.grid(True)
st.pyplot(plt)
# Clears the current figure
plt.clf()


st.write('This histogram shows the distribution of laptop prices.')
st.write('**Observation:** Most laptops are in the 500 - 1,000 euro price point, with only some high-end laptops in the 1,500 - 3,000 euro range, and only a miniscule amount costing 3,000 euros upward.')
st.header('------------------------------------------------------------')

#Graph10
st.header('Box Plot: Price Comparison of Asus, Lenovo, and Dell Laptops by CPU Company')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/mnt/data/laptop_price-dataset.csv'
laptop_data = pd.read_csv('laptop_price-dataset.csv')

# Filter the dataset to focus only on Asus, Lenovo, and Dell
filtered_data = laptop_data[laptop_data['Company'].isin(['Asus', 'Lenovo', 'Dell'])]

# Create the boxplot with three variables (Company, CPU Company, and Price)
plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_data, x='Company', y='Price (Euro)', hue='CPU_Company', palette='Set3')

# Set plot title and labels
plt.title('Box Plot: Price Comparison of Asus, Lenovo, and Dell Laptops by CPU Company')
plt.xticks(rotation=45)
st.pyplot(plt)
# Clears the current figure
plt.clf()


st.write('This is a Box Plot of Price Comparison of Asus, Lenovo, and Dell Laptops by CPU Company')
st.write('**Observation:** Based on the box plot, Asus offers Intel and AMD laptops that are both in a similar price range. These laptops are priced 500 - 2000, however, Asus’ Intel laptops can be slightly pricier than their AMD laptops. It is also noted that they also offer some Intel laptops that are an outlier, being priced around 4000 euros. Dell mostly offers Intel-based laptops. These laptops are priced around 1000 - 3000 euros. Like Asus, they also have some outliers that are pricier, being above 3000 euros. It is to be noted that there aren’t any AMD-based laptops in the dataset. Lenovo also offers both Intel and AMD-based laptops. Their Intel laptops are more varied in price ranges and can be priced higher than their AMD laptops which are mostly below 1000 euros. In addition, their Intel laptops are priced around 1000 to 3000 euros with some going above the mentioned price range.')
st.header('------------------------------------------------------------')

st.header('Conclusion')
st.subheader('**1. Price Distribution:**')
st.write('*   Laptops typically cost around €1134.97.')
st.write('*   Range of prices: At least: €174, Maximum: €6099, '
          '25% of the total: €609, and 75% of the total: €1496.50')

st.subheader('**2. RAM Distribution and Relation to Price:**')
st.write('*   RAM allocation: 48% of laptops have 8 GB, '
          '4 GB: 28.8%, and 16 GB: 15.5%')
st.write('*   Cost determined by RAM: At €3975, laptops with 64 GB '
          'of RAM are the most expensive on average. Laptops with 8 GB RAM '
          'often cost €1184, whilst laptops with 4 GB RAM typically cost €576.')

st.subheader('**3. Company Distribution and Price:**')
st.write('*   Leading brands: Dell: 22.8%, Lenovo: 22.7%, and HP: 21%')
st.write('*   Cost according to brand: Razer laptops are the most expensive on average '
          'at €3346, while Apple laptops are more expensive on average at €1564. '
          'The average cost of an Acer laptop is less, at €633.')

st.subheader('**4. Screen Size Distribution:**')
st.write('*   Most laptop screens are between 14 and 15.6 inches in size, '
          'with an average of 15 inches. The range of screen sizes is 10.1 inches to 18.4 inches.')

st.subheader('**5. Operating System Distribution and Price:**')
st.write('*   OS allocation: With 82.2% of laptops, Windows 10 is widely used, '
          'No OS: 5.2%, and Linux: 4.5%')
st.write('*   Cost according to OS: MacOS laptops are among the priciest, '
          'with an average cost of €1749.63. The average cost of a Windows 7 laptop '
          'is €1686.65. Chrome OS computers are less expensive, with an average '
          'price of €553.59.')
