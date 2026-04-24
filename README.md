# Right to Repair Movement Data Analysis - Caleb Mastruserio

## Project Overview

This project analyzes the increasing cost of vehicle repairs and explores how the **Right to Repair Movement** could help reduce these costs. The analysis compares repair costs across different sources, including dealership-level data, consumer price trends, and independent shop pricing datasets.

The goal is to demonstrate that modern vehicles, which rely heavily on proprietary software and complex systems, are more expensive to repair and restrict consumer choice.


## Research Question

How much more expensive are dealership repairs compared to independent shops, and what role can the Right to Repair Movement play in reducing these costs?


## Project Structure
Final-Project-CALEBMASTRUSERIO/
│
├── README.md
|
├── Report/
│   └── Final_Report.docx (or PDF)
│
├── 2022CarMD.csv
├── CPE.csv
│
├── Python/
│   └── analysis.ipynb
│
│
├── PowerBI/
│   └── dashboard.pbix
|
├── assets/
│ └── banner.jpeg



## Data Sources

1. **CarMD Vehicle Health Index**  
   Provides repair cost data across different vehicles and regions  
   https://carmd.com/pages/vehicle-health-index  

2. **Consumer Price Index (FRED)**  
   Tracks vehicle maintenance and repair cost trends over time  
   https://fred.stlouisfed.org/series/CUSR0000SETD  

3. **Car Parts Price Estimation Dataset (Kaggle)**  
   Provides part-level pricing across different repair shops  
   https://www.kaggle.com/datasets/shorooq77/car-parts-price-estimation  




## How to Run the Project

### 1. Install Required Libraries

pip install pandas numpy matplotlib seaborn scikit-learn

### 2. All of the data files are included

### 3. Notebook is ready to be ran




## Data Cleaning Steps

-Removed duplicate records

-Handled missing values using removal or imputation

-Converted price columns from strings to numeric values

-Standardized column names for consistency


## Key Visualizations

-Histogram of repair cost distribution

-Boxplot to identify outliers

-Bar chart of average part prices across shops

-Line chart comparing shop pricing differences


## Machine Learning Approach

-Used Linear Regression to predict repair costs

-Applied One-Hot Encoding for categorical variables

-Split data into training and testing sets (80/20)

-Evaluated model predictions on unseen data


## Key Findings

-Repair costs have increased significantly over time

-Newer vehicles are more expensive to repair due to software complexity

-Independent shop pricing shows consistent and predictable patterns

-Data suggests repairs do not require exclusive dealership access


## Conclusion

The analysis supports the idea that rising repair costs are driven not only by inflation but by increased technological complexity and restricted access to repair tools. The Right to Repair Movement can help reduce these costs by enabling greater access to repair data and tools for independent mechanics and consumers.
