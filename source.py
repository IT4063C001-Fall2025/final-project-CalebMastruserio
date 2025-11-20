#!/usr/bin/env python
# coding: utf-8

# # Right to Repair Movement Analysis📝
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 
# The social issue I am trying to address is the the Right to Repair Movement. The growing probelm at hand is in the modern construction and design of new vehciles today, being made so the average mechanic or even DIYer cannot work on or repair their own car without taking it to the manufacturer for speaclized work. By is a growing problem in the modern automotive industry as newer cars become more and more complex and complicated to work on and repair. With modules upon modules to service and scan that require dealer-only diagnostic equipment and software, this prevents many car owners from DIYers to small independent shops from servicing and working on their cars without taking them to the expensive dealership. 

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 
# The question I am seeking to answer is how much more expensive are dealerships in conjucnction with the growing requirement of proprietory software in cars compare to in cost of independent shops or even DIYers and what role can the Right to Repair Movement play in reducing those costs?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 
# The average cost of modern vehicles and repairs of newer cars have increase significantly. With the technological shift from mechanically driven, simple, and easy to work on cars that were once software dependent, have raised modern day repair costs and resticted consumer choice. The right to repair movement offers a solution to the growing complexity of modern construction automobiles by leveling the grounds between the big name dealerships and independent shops and DIYers, reducing the costs of repairs and giving vehicle owners a choice in where to service their vehicles. 

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 📝 <!-- Answer Below -->

# 1. The first data source I am going to be using is the Vehicle Health Index by CarMD. This is going to provide me with information to show the costs of over older VS newer vehicles and their associated repair costs which shows how software dependencies can raise repair prices. I will relate this data source because it provides me with a breakdown in measuring the average annual repair costs among varying categories of repair and shows how the costs of automotive repair have increased over time. 
# 
# 
# 2. The second data source I will use is the Consumer Price Index for All Urban Consumers: Motor Vehicle Maintenance and Repair in U.S. City Average. This is going to show me the cost of vehicle maitenence by year to display the rising costs of automotive repair costs. This will be related by displaying with direct visual information how the costs of vehicle maitenence has increased throughout the years. 
# 
# 3. The final data source i will be using is a database provided from kaggle.com, the "Car Parts Price Estimation Dataset." This dataset will provide me with an example cost breakdown of the prices of commmon parts found to repair a Toyota Camry. This will display the increase in cost over a span of a few years. 

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 
# The approach I have is to combine the cost data, and vehicle data to build a full picture of how repair costs have changed over time. I will demonstrate how repair costs have changed between older cars and newers cars over the last few years. Through the use of repair costs datasets I will measure how repair costs differ by the year and type of repairs performed. This will directly demonstrate how modern vehicles are more expensive to repair as compared to simpler and less complicated older vehicle models. 

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[ ]:


https://carmd.com/pages/vehicle-health-index

https://fred.stlouisfed.org/series/CUSR0000SETD

https://www.kaggle.com/datasets/shorooq77/car-parts-price-estimation


# ## 1. Exploratory Data Analysis (EDA)
# You may have (hopefully) found more datasets for your project. If so, you should include them under the "Data Sources" section. If not, that means that so far your data is able to answer your analysis Apply EDA and use Visualizations to answer questions such as:
# 
# What insights and interesting information are you able to extract at this stage?
# Currently, as my question has posed, vehicle repair costs have steadily increased over time. Most espeacially for newer cars with more software and computer dependent systems and features. Electric and diagnostic related fixes are the main driving focus of this trend, while inflation alone does not completely explain the durastic increase overtime in these expenses. These shown trends show that modern vehicles and their overly reliant designs on technology restrict repair access and are contributing to higher overall maintenance costs. 
# 
# What are the distributions of my variables?
# Repair cost and parts prices are both skewed to the right, having most repair bills fall within a moderate price range, and a lower amount containing very high cost electric or software related repairs pulling the average price trend upwards. The CPI data shows a steady upward trend, until you get later into the more recent years, such as the last 5 years, and the prices for maintenance costs have jumped upwards strongly. 
# 
# Are there any correlations between my variables?
# There is a moderate positive correlation between vehicle year and average repair cost, meaning that newer cars tend to cost more to repair and are more expensive to own and operate. This supports the idea that modern cars with complex software and electronics significantly increase repair costs compared to older vehicles with more simpler components, features, and systems. 
# 
# What issues can you see in your data at this point?
# So far I have noticed that some of the datasets contain missing or duplicate values and a couple of the columns and even a few columns are stored as text insteda of numbers. Additionally, there are outliers in part prices and repair bills. Likely to be rare events or even a complex dealer component. This could distort averages and data visualizations. I will go through many of these issues before deeper data analysis can occur. 
# 
# Are there any outliers or anomalies? are they relevant to your analysis? or should they be removed?
# Yes so far in my playing with the data I have seen a few high cost outliers as I mentioned in the previous question. Such as some exsorbant costs that do not reflect a direct trend or indicate a large jump in cost, but could pose a challenge when attempting to provide deeper data analysis. I will do more research in regards to how but it will affect my data and/or visualizations. 
# 
# Are there any missing values? how are you going to deal with them?
# Yes, I have noticed that the Car Price Estimation dataset contains some missing values, such as listing the same parts for specific model years. Those rows will be removed and since they represent only a small amount of the data and are very unlikely to affect the overall trend of data. This will allow me to conduct a linear analysis of the parts costs by year. 
# 
# Are there any duplicate values? how are you going to deal with them?
# Yes, as I have played around with the data, I have noticed as well as in the Car Price Estimation dataset that it does contain dupelicate records for the same exact part and year. Similar to the missing values, I can remove them and still have a clear visual of the averages ensuring they are not skewed. 
# 
# 
# Are there any data types that need to be changed?
# Yes, only a few columns contain prices in both the CarMD and the Car Prices Estimate datasets are stored with strings as "$" symbols. I will go ahead and convert them to numerical values so that I can conduct more accurate and reflective calculations and visualizations. 
# 
# 
# ## Data Visualization
# You should use several visualizations to help answer the questions above.
# 
# You should have at least 4 visualizations in your notebook, to represent different aspects and valuable insights of your data.
# 
# You can use 2 visualization library that you want.
# 
# You can use any type of visualization that best represents your data.
# 
# You'll need to provide a short description of each visualization and explain what it represents, and what insights you can extract from it.
# 

# In[ ]:


import pandas as pd

df = pd.read_csv(r"c:\Users\kid0001\Documents\University of Cincinnati\Fall Semester 2025\DATA TECH ANALYTICS\Final Project Stuff\2022CarMD.csv")

print(df.head())
print(df.info())


# In[5]:


df.columns = df.columns.str.replace(" ", "_").str.replace(r"[^\w]", "", regex=True)
print(df.columns)


# In[8]:


numeric_cols = ['AvgLabor_Cost', 'Avg_Parts_Cost', 'Avg_Repair_Costs_Parts__Labor']

for col in numeric_cols:
    df[col] = df[col].replace('[\$,]', '', regex=True).astype(float)

print(df.head())


# In[ ]:


# Missing value
print("Missing values:\n", df.isnull().sum())

# dupelicate value
print("Duplicate rows:", df.duplicated().sum())


# In[ ]:


## Visual 1: Histogram – Distribution of Total Repair Costs
# This is what have used to show the typical range VS higher cost outliers. 

plt.figure(figsize=(8,5))
sns.histplot(df['Avg_Repair_Costs_Parts__Labor'], bins=15, kde=True, color='skyblue')
plt.title("Distribution of Average Repair Costs Across States – 2022")
plt.xlabel("Average Repair Cost ($)")
plt.ylabel("Number of States")
plt.show()


# In[11]:


## Visual 2: Boxplot – Identify Outliers
# This is part of what I have used to display some of the potential outliers for my datasets. 
# Showing my mean, quartiles, and any extremely outliers.

plt.figure(figsize=(8,5))
sns.boxplot(y=df['Avg_Repair_Costs_Parts__Labor'], color='lightgreen')
plt.title("Boxplot of Average Repair Costs – 2022")
plt.ylabel("Average Repair Cost ($)")
plt.show()


# In[ ]:


import pandas as pd

car_parts = pd.read_csv(r"C:\Users\kid0001\Documents\University of Cincinnati\Fall Semester 2025\DATA TECH ANALYTICS\Final Project Stuff\CPE.csv", encoding='latin-1')

print(car_parts.head())
print(car_parts.info())


# In[ ]:


# Cleaning column names
car_parts.columns = car_parts.columns.str.replace(" ", "_").str.replace(r"[^\w]", "", regex=True)
print("Cleaned columns:", car_parts.columns)


# In[ ]:


#More cleanup
import numpy as np

price_cols = ['shop1_eqbal', 'shop2haraj', 'shop3_']

for col in price_cols:

    car_parts[col] = pd.to_numeric(car_parts[col].replace('[\$,]', '', regex=True), errors='coerce')


print(car_parts[price_cols].head())


# In[ ]:


import numpy as np

price_cols = ['shop1_eqbal', 'shop2haraj', 'shop3_']
for col in price_cols:
    car_parts[col] = pd.to_numeric(car_parts[col].replace('[\$,]', '', regex=True), errors='coerce')

car_parts['avg_price'] = car_parts[price_cols].mean(axis=1)

print(car_parts[['body_part', 'avg_price']].head())


# In[ ]:


## Visual 3: CPE bar chart.
# This bar chart shows that some car parts are a lot more expensive across the shops, supporting the idea that more modern software dependent vehicle repairs increase the repair costs. 

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,6))
sns.barplot(data=car_parts, x='body_part', y='avg_price', palette="Blues_r")
plt.xticks(rotation=45)
plt.xlabel("Car Part")
plt.ylabel("Average Price ($)")
plt.title("Average Price of Car Parts Across Shops")
plt.show()


# In[ ]:


## Visual 4: Line Chart of Price comparisons
# This chart shows the prices for the same parts and how they vary significantly accross the shops displaying the fact of how propriertory systems drive up repair costs. 
plt.figure(figsize=(12,6))
for col in price_cols:
    sns.lineplot(data=car_parts, x='body_part', y=col, marker='o', label=col)

plt.xticks(rotation=45)
plt.xlabel("Car Part")
plt.ylabel("Price ($)")
plt.title("Comparison of Part Prices Across Different Shops")
plt.legend(title='Shop')
plt.grid(True)
plt.show()


# ## 2. Data Cleaning and Transformation
# In this section, you'll clean data per your findings in the EDA section. You will be handling issues such as:
# 
# Missing values
# 
# Duplicate values
# 
# Anomalies and Outliers
# 
# Data types transformation.
# 
# 
# You will need to describe the data cleaning process you went through to prepare your data for analysis. This includes your code and the reasoning behind your decisions.
# 
# See above the code I used to detect for any missing, dupelicates, outliers, and data types for transformations. I started by cleaning up the data by converting non-numeric values to NAN, dropping any uncecessary columns, transforming price columns to numeric types, creating an average price column, and keeping high cost outliers because they were relevant to showing how the more modern build a vehicle is has a strong positive correllation to increased part and labor bills. 
# 

# ## 3. Prior Feedback and Updates & Machine Learning 
# What feedback did you receive from your peers and/or the teaching team?
# 
# What changes have you made to your project based on this feedback?
# 
# I did not recieve too much feedback from my peers, I just spent this time on refining my data cleansing, and making sure that my datasets were ready for analysis and practicing how I can use the data to make visualizations to support my findings on my project question. 

# Machine Learning Plan
# I plan on using supervised learning, specifically regression models to predict repair costs based on factors like car model, part type, and year. Some potential issues I might encounter are limited and incomplete data. I have already seen this in trying to find good data sets to use for my analysis. As well as inconsistent prices across shops, which pose the risk of model accuracy. Some challenges have already began to surface, such as cleaning and standardizing the data. This is something I am still trying to learn as we go, so its given me some trouble. Finally, handling any outliers, and ensuring the model generalizes well to different parts and types of cars will be a fun feat to tackle. 
# 
# 

# ### Machine Learning Model Implementation

# 

# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Features and target
X = ml_data[['car', 'model', 'body_part']]
y = ml_data['avg_price']

# Categorical columns
cat_cols = ['car', 'model', 'body_part']

# Preprocessing: convert text → numeric
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

# Build pipeline
model = Pipeline([
    ('prep', preprocess),
    ('reg', LinearRegression())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

print("Model trained successfully!")


# In[10]:


from sklearn.metrics import mean_squared_error
import numpy as np

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("RMSE:", rmse)


# In[11]:


def predict_repair_cost(car, model_name, body_part):
    new_data = pd.DataFrame({
        'car': [car],
        'model': [model_name],
        'body_part': [body_part]
    })

    prediction = model.predict(new_data)[0]
    return round(prediction, 2)

# Example:
print(predict_repair_cost("Toyota", "Corolla", "Front Bumper"))


# In[16]:


import seaborn as sns

pred_df = X_test.copy()
pred_df['y_pred'] = y_pred

plt.figure(figsize=(12,6))
sns.boxplot(x='body_part', y='y_pred', data=pred_df)
plt.xticks(rotation=45)
plt.xlabel("Car Part")
plt.ylabel("Predicted Repair Cost")
plt.title("Predicted Repair Costs by Car Part")
plt.show()


# In[26]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv("C:/Users/kid0001/Documents/University of Cincinnati/Fall Semester 2025/DATA TECH ANALYTICS/Final Project Stuff/CPE.csv", encoding='latin1')

data = data.replace({'null': None})
for col in ['shop1 (eqbal)', 'shop2(haraj)', 'shop3 ']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

data = data.dropna(subset=['shop1 (eqbal)', 'shop2(haraj)', 'shop3 '])

X = data[['shop1 (eqbal)', 'shop2(haraj)']]
y = data['shop3 ']

model = LinearRegression()
model.fit(X, y)

shop1_range = np.linspace(X['shop1 (eqbal)'].min(), X['shop1 (eqbal)'].max(), 20)
shop2_range = np.linspace(X['shop2(haraj)'].min(), X['shop2(haraj)'].max(), 20)
shop1_grid, shop2_grid = np.meshgrid(shop1_range, shop2_range)
shop3_pred = model.predict(np.c_[shop1_grid.ravel(), shop2_grid.ravel()]).reshape(shop1_grid.shape)

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X['shop1 (eqbal)'], X['shop2(haraj)'], y, color='blue', label='Actual')

ax.plot_surface(shop1_grid, shop2_grid, shop3_pred, color='orange', alpha=0.5, label='Regression Plane')

ax.set_xlabel('shop1 (eqbal)')
ax.set_ylabel('shop2(haraj)')
ax.set_zlabel('shop3')
ax.set_title('3D Regression Plane')
plt.show()


# The regression model supports my claim about the Right to Repair Movement by demonstrating that data from independent shops can predict repair outcomes at other shops. This suggests that knowledge about vehicle repairs is not exclusive to dealerships and specialized diagnostic tools. The model shows that patterns in repair costs and processes exist outside manufacturer-controlled systems, meaning independent mechanics and DIYers could feasibly perform repairs if they had access to the right information. This aligns with the Right to Repair philosophy, highlighting that barriers preventing independent repairs are not absolute and can be challenged with accessible data.

# ## Data Splitting

# In[ ]:


# Here I list the numeric columns and convert any to numeric if they are not already, for data cleaning. 
numeric_cols = ['shop1 (eqbal)', 'shop2(haraj)', 'shop3 ']

for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')



# In[ ]:


# Here we can predict one shop's cost from another. This is used to show common trends and accurately see the overall similarities in cost for the different components.
X = data[['shop1 (eqbal)', 'shop2(haraj)']]

y = data['shop3 ']


# In[ ]:


# Here we are using a split to separate into training data to evaluate how well it generalizes. 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[ ]:


# This step we are going to drop any rows and/or features where the target is NaN, then we split and target again. 
clean_data = data.dropna(subset=['shop1 (eqbal)', 'shop2(haraj)', 'shop3 '])

X = clean_data[['shop1 (eqbal)', 'shop2(haraj)']]
y = clean_data['shop3 ']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[ ]:


# Now we are loading another CSV for data cleaning and separation. Following similar steps as shown from above. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

data = pd.read_csv(
    "C:/Users/kid0001/Documents/University of Cincinnati/Fall Semester 2025/DATA TECH ANALYTICS/Final Project Stuff/CPE.csv",
    encoding='latin1' 
)

numeric_cols = ['shop1 (eqbal)', 'shop2(haraj)', 'shop3 ']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

X = data[['shop1 (eqbal)', 'shop2(haraj)']]
y = data['shop3 ']

imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
y_imputed = y.fillna(y.mean())



# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y_imputed, test_size=0.2, random_state=42
)


# In[ ]:


model = LinearRegression()
model.fit(X_train, y_train) 



# In[ ]:


# This step we can use our test data to predict costs for shop3. 
y_pred = model.predict(X_test)
print(y_pred[:5]) 


# In[1]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

