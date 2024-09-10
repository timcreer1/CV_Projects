#import
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
pd.options.display.max_colwidth = 200

df = pd.read_csv('/Users/creer/PycharmProjects/CV_Projects/.venv/Predicting Purchases BFS/Data/BFS.csv')
df.head()

# Displaying DataFrame info
print("DataFrame Info:")
df.info()  # This directly prints the DataFrame information

# Displaying DataFrame description
print("\nDataFrame Description:")
print(df.describe())  # Use print to display the DataFrame description in scripts

# Displaying count of missing values
print("\nMissing Values Count:")
print(df.isnull().sum().to_frame(name='Missing Values'))  # Prints missing values count in each column

# Displaying count of unique values per column
print("\nUnique Values Count:")
print(df.nunique().to_frame(name='Unique Values'))  # Prints the count of unique values for each column


# 3. Data Exploration

plt.figure(figsize=(10, 3))
sns.histplot(df.Purchase, kde=True, palette="YlOrBr")
plt.title("Distribution of Purchase", fontsize=15)
plt.xlabel("Skewness -->  {:.2f}, Kurtosis --> {:.2f}".format(df.Purchase.skew(), df.Purchase.kurtosis()));
plt.show()

# Calculate the IQR and the outliers
plt.figure(figsize=(10, 2))
q1 = df['Purchase'].quantile(0.25)
q3 = df['Purchase'].quantile(0.75)
iqr = q3 - q1
outliers = df[(df['Purchase'] < (q1 - 1.5 * iqr)) | (df['Purchase'] > (q3 + 1.5 * iqr))]
num_outliers = len(outliers)
total_data = len(df)

# Create the box plot
sns.boxplot(x=df['Purchase'], color="red", orient='h')

# Add the IQR and the percentage of outliers to the plot
plt.text(q3 + iqr * 0.5, -0.25, "IQR: {:.2f}".format(iqr), ha='center')
plt.text(q3 + iqr * 0.5, -0.35, "Outliers: {:.2f}%".format(num_outliers / total_data * 100), ha='center')
plt.show()

# Calculate the total purchase amount for each gender
total_purchase = df.groupby('Gender')['Purchase'].sum().reset_index()
print('total_purchase')
print(total_purchase)
print('\n')

# Calculate the average purchase amount for each gender
average_purchase = df.groupby('Gender')['Purchase'].mean().reset_index()
print('average_purchase')
print(average_purchase)
print('\n')
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# First graph: Pie chart showing the distribution of male and females in the gender column
plt.subplot(131)
df['Gender'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Gender Distribution")

# Second graph: Bar chart showing total purchase amount per gender
plt.subplot(132)
sns.barplot(x='Gender', y='Purchase', data=total_purchase)
plt.title("Total Purchase Amount per Gender")

# Third graph: Bar chart showing average purchase amount per gender
plt.subplot(133)
sns.barplot(x='Gender', y='Purchase', data=average_purchase)
plt.title("Average Purchase Amount per Gender")
plt.show()

# Calculate the total purchase amount for each gender
total_purchase = df.groupby('Marital_Status')['Purchase'].sum().reset_index()
print('total_purchase')
print(total_purchase)
print('\n')

# Calculate the average purchase amount for each gender
average_purchase = df.groupby('Marital_Status')['Purchase'].mean().reset_index()
print('average_purchase')
print(average_purchase)
print('\n')
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# First graph: Pie chart showing the distribution of male and females in the gender column
plt.subplot(131)
df['Marital_Status'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Gender Distribution")

# Second graph: Bar chart showing total purchase amount per gender
plt.subplot(132)
sns.barplot(x='Marital_Status', y='Purchase', data=total_purchase, palette='hls')
plt.title("Total Purchase Amount per Marital_Status")

# Third graph: Bar chart showing average purchase amount per gender
plt.subplot(133)
sns.barplot(x='Marital_Status', y='Purchase', data=average_purchase, palette='hls')
plt.title("Average Purchase Amount per Marital_Status")
plt.show()

# Calculate the total purchase amount for each gender
total_purchase = df.groupby('Occupation')['Purchase'].sum().reset_index()
print('total_purchase')
print(total_purchase)
print('\n')

# Calculate the average purchase amount for each gender
average_purchase = df.groupby('Occupation')['Purchase'].mean().reset_index()
print('average_purchase')
print(average_purchase)
print('\n')
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# First graph: Pie chart showing the distribution of male and females in the gender column
plt.subplot(131)
df['Occupation'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Occupation Distribution")

# Second graph: Bar chart showing total purchase amount per gender
plt.subplot(132)
sns.barplot(x='Occupation', y='Purchase', data=total_purchase, palette='hls')
plt.title("Total Purchase per Occupation")

# Third graph: Bar chart showing average purchase amount per gender
plt.subplot(133)
sns.barplot(x='Occupation', y='Purchase', data=average_purchase, palette='hls')
plt.title("Average Purchase per Occupation")
plt.show()

# Calculate the total purchase amount for each gender\
df['Stay_In_Current_City_Years'].value_counts()
total_purchase = df.groupby('Stay_In_Current_City_Years')['Purchase'].sum().reset_index()
print('total_purchase')
print(total_purchase)
print('\n')

# Calculate the average purchase amount for each gender
average_purchase = df.groupby('Stay_In_Current_City_Years')['Purchase'].mean().reset_index()
print('average_purchase')
print(average_purchase)
print('\n')
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# First graph: Pie chart showing the distribution of male and females in the gender column
plt.subplot(131)
df['Stay_In_Current_City_Years'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Stay_In_Current_City_Years Distribution")

# Second graph: Bar chart showing total purchase amount per gender
plt.subplot(132)
sns.barplot(x='Stay_In_Current_City_Years', y='Purchase', data=total_purchase, palette='hls')
plt.title("Total Purchase per Stay_In_Current_City_Years")

# Third graph: Bar chart showing average purchase amount per gender
plt.subplot(133)
sns.barplot(x='Stay_In_Current_City_Years', y='Purchase', data=average_purchase, palette='hls')
plt.title("Average Purchase per Stay_In_Current_City_Years")
plt.show()

# Calculate the total purchase amount for each gender
total_purchase = df.groupby('Age')['Purchase'].sum().reset_index()
print('total_purchase')
print(total_purchase)
print('\n')

# Calculate the average purchase amount for each gender
average_purchase = df.groupby('Age')['Purchase'].mean().reset_index()
print('average_purchase')
print(average_purchase)
print('\n')
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# First graph: Pie chart showing the distribution of male and females in the gender column
plt.subplot(131)
df['Age'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Age Distribution")

# Second graph: Bar chart showing total purchase amount per gender
plt.subplot(132)
sns.barplot(x='Age', y='Purchase', data=total_purchase, palette='hls')
plt.title("Total Purchase per Age")

# Third graph: Bar chart showing average purchase amount per gender
plt.subplot(133)
sns.barplot(x='Age', y='Purchase', data=average_purchase, palette='hls')
plt.title("Average Purchase per Age")
plt.show()