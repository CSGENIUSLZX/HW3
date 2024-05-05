# PPHA 30537
# Spring 2024
# Homework 3

# YOUR NAME HERE
#zixuanlan
# YOUR CANVAS NAME HERE
#zixuanlan
# YOUR GITHUB USER NAME HERE
#csgeniuslzx
# Due date: Sunday May 5th before midnight
# Write your answers in the space between the questions, and commit/push only
# this file to your repo. Note that there can be a difference between giving a
# "minimally" right answer, and a really good answer, so it can pay to put
# thought into your work.

##################

#NOTE: All of the plots the questions ask for should be saved and committed to
# your repo under the name "q1_1_plot.png" (for 1.1), "q1_2_plot.png" (for 1.2),
# etc. using fig.savefig. If a question calls for more than one plot, name them
# "q1_1a_plot.png", "q1_1b_plot.png",  etc.

# Question 1.1: With the x and y values below, create a plot using only Matplotlib.
# You should plot y1 as a scatter plot and y2 as a line, using different colors
# and a legend.  You can name the data simply "y1" and "y2".  Make sure the
# axis tick labels are legible.  Add a title that reads "HW3 Q1.1".

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
os.chdir('C:/Users/62473/Desktop/Python Data/HW3/homework-3-CSGENIUSLZX-main')
x = pd.date_range(start='1990/1/1', end='1991/12/1', freq='MS')
y1 = np.random.normal(10, 2, len(x))
y2 = [np.sin(v)+10 for v in range(len(x))]

plt.figure(figsize = (10,10),dpi=80)
plt.plot(x, y2, color = 'red', label = 'y2')
plt.scatter(x, y1, color = 'blue',label='y1')
plt.legend()
plt.xlabel('date')
plt.ylabel('random number')
plt.title('HW3 Q1.1')
plt.show()
# Question 1.2: Using only Matplotlib, reproduce the figure in this repo named
# question_2_figure.png.
x=list(range(10,21))
y1=list(range(10,21))
y2=list(range(20,9,-1))
plt.plot(x, y1, color = 'blue',label='blue')
plt.plot(x, y2,color = 'red',label='red')
plt.legend(loc='center left')
plt.title('X marks the spot')
plt.xticks(range(10, 19, 2))
plt.yticks(range(10, 19, 2))
plt.show()
# Question 1.3: Load the mpg.csv file that is in this repo, and create a
# plot that tests the following hypothesis: a car with an engine that has
# a higher displacement (i.e. is bigger) will get worse gas mileage than
# one that has a smaller displacement.  Test the same hypothesis for mpg
# against horsepower and weight.
df=pd.read_csv('mpg.csv')
pd.set_option('display.max_columns', None)
df.head(5)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

plt.figure(figsize=(10, 6))
sns.scatterplot(x='displacement', y='mpg', data=df,ax=axes[0])
axes[0].set_title('Displacement vs. MPG')
axes[0].set_xlabel('Displacement (cc)')
axes[0].set_ylabel('Miles per Gallon')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='horsepower', y='mpg', data=df,ax=axes[1])
axes[1].set_title('Horsepower vs. MPG')
axes[1].set_xlabel('Horsepower')
axes[1].set_ylabel('Miles per Gallon')


plt.figure(figsize=(10, 6))
sns.scatterplot(x='weight', y='mpg', data=df,ax=axes[2])
axes[2].set_title('Weight vs. MPG')
axes[2].set_xlabel('Weight')
axes[2].set_ylabel('Miles per Gallon')


plt.tight_layout()

plt.show()
# Question 1.4: Continuing with the data from question 1.3, create a scatter plot 
# with mpg on the y-axis and cylinders on the x-axis.  Explain what is wrong 
# with this plot with a 1-2 line comment.  Now create a box plot using Seaborn
# that uses cylinders as the groupings on the x-axis, and mpg as the values
# up the y-axis.
plt.figure(figsize=(10, 10))
plt.scatter(x='cylinders', y='mpg',data=df)
plt.show()
#The data points in the plot look very dense, especially at some specific x values, and we chose the wrong image

sns.boxplot(x='cylinders', y='mpg',data=df)
plt.show()
# Question 1.5: Continuing with the data from question 1.3, create a two-by-two 
# grid of subplots, where each one has mpg on the y-axis and one of 
# displacement, horsepower, weight, and acceleration on the x-axis.  To clean 
# up this plot:
#   - Remove the y-axis tick labels (the values) on the right two subplots - 
#     the scale of the ticks will already be aligned because the mpg values 
#     are the same in all axis.  
#   - Add a title to the figure (not the subplots) that reads "Changes in MPG"
#   - Add a y-label to the figure (not the subplots) that says "mpg"
#   - Add an x-label to each subplot for the x values
# Finally, use the savefig method to save this figure to your repo.  If any
# labels or values overlap other chart elements, go back and adjust spacing.
fig,ax=plt.subplots(2,2,figsize=(10,10))
fig.subplots_adjust(hspace=0.3,wspace=0.3)

sns.scatterplot(x='displacement', y='mpg', data=df, ax=ax[0, 0])
ax[0, 0].set_title('Displacement')
ax[0, 0].set_xlabel('Displacement (cc)')
ax[0, 0].set_ylabel('mpg')

sns.scatterplot(x='horsepower', y='mpg', data=df, ax=ax[0, 1])
ax[0, 1].set_title('Horsepower')
ax[0, 1].set_xlabel('Horsepower')
ax[0, 1].set_ylabel('')
ax[0, 1].set_yticklabels([])

sns.scatterplot(x='weight', y='mpg', data=df, ax=ax[1, 0])
ax[1, 0].set_title('Weight')
ax[1, 0].set_xlabel('Weight')
ax[1, 0].set_ylabel('mpg')

sns.scatterplot(x='acceleration', y='mpg', data=df, ax=ax[1, 1])
ax[1, 1].set_title('Acceleration')
ax[1, 1].set_xlabel('Acceleration')
ax[1, 1].set_ylabel('')
ax[1, 1].set_yticklabels([])
fig.suptitle('Changes in MPG', fontsize=16)
fig.text(0.04, 0.5, 'Miles Per Gallon (mpg)', va='center', rotation='vertical', fontsize=12)
plt.savefig('question1.5.png')
plt.show()

# Question 1.6: Are cars from the USA, Japan, or Europe the least fuel
# efficient, on average?  Answer this with a plot and a one-line comment.
sns.boxplot(x='origin', y='mpg', data=df)
plt.show()
#USA'S cars are the worst,are the worst fuel efficient
# Question 1.7: Using Seaborn, create a scatter plot of mpg versus displacement,
# while showing dots as different colors depending on the country of origin.
# Explain in a one-line comment what this plot says about the results of 
# question 1.6.

plt.figure(figsize=(10, 6))
sns.scatterplot(x='displacement', y='mpg', hue='origin', data=df, palette='bright')
plt.title('MPG vs. Displacement Colored by Origin')
plt.xlabel('Displacement (cc)')
plt.ylabel('Miles per Gallon (MPG)')
plt.show()
#USA CARS have higher displacement and lower mpg on average,so they are the lesast fuel efficient

# Question 2: The file unemp.csv contains the monthly seasonally-adjusted unemployment
# rates for US states from January 2020 to December 2022. Load it as a dataframe, as well
# as the data from the policy_uncertainty.xlsx file from homework 2 (you do not have to make
# any of the changes to this data that were part of HW2, unless you need to in order to 
# answer the following questions).
df_1=pd.read_excel('policy_uncertainty.xlsx')
df_2=pd.read_csv('unemp.csv')
df_2.head(5)
df_1.head(5)
df_2.describe()
df_1['DATE'] = pd.to_datetime(df_1[['year', 'month']].assign(day=1))
df_2['DATE'] = pd.to_datetime(df_2['DATE'])
#    2.1: Merge both dataframes together
merged_df = pd.merge(df_1, df_2, on='DATE', how='inner')
merged_df.head(5)
#    2.2: Calculate the log-first-difference (LFD) of the EPU-C data
merged_df['EPU_National_log'] = np.log(merged_df['EPU_National']+ 1e-10)
merged_df['EPU_National_LFD'] = merged_df['EPU_National_log'].diff()
merged_df.dropna(inplace=True)
#    2.2: Select five states and create one Matplotlib figure that shows the unemployment rate
#         and the LFD of EPU-C over time for each state. Save the figure and commit it with 
#         your code.
states = ['CA', 'TX', 'NY', 'FL', 'IL']  # 示例州名

plt.figure(figsize=(15, 10))

for state in states:
    subset = merged_df[merged_df['STATE'] == state]
    plt.plot(subset['DATE'], subset['unemp_rate'], label=f'{state} Unemployment Rate')
    plt.plot(subset['DATE'], subset['EPU_National_LFD'], label=f'{state} EPU National LFD', linestyle='--')

plt.title('Unemployment Rate and Log-First-Difference of EPU-National Over Time')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()

plt.savefig('state_unemployment_epu.png')
plt.show()


#    2.3: Using statsmodels, regress the unemployment rate on the LFD of EPU-C and fixed
#         effects for states. Include an intercept.
import statsmodels.api as sm

# 回归分析：EPU的LFD对失业率的影响

merged_df.dropna(inplace=True)
X = merged_df[['EPU_National_LFD']]  # 自变量
X = sm.add_constant(X)  # 添加常数项
y = merged_df['unemp_rate']  # 因变量

model = sm.OLS(y, X).fit()
results_summary = model.summary()
#    2.4: Print the summary of the results, and write a 1-3 line comment explaining the basic
#         interpretation of the results (e.g. coefficient, p-value, r-squared), the way you 
#         might in an abstract.
print(results_summary)
#The intercept of the model is about 5.1809, the standard error is very small (0.010),
# and it is highly statistically significant (p-value < 0.001),
# indicating that without the influence of any other variables,
# the average unemployment rate is 5.1809%.
#The EPU_National_LFD coefficient is -0.0003, indicating that there is a negative correlation between the logarithmic first difference (LFD) of EPU and the unemployment rate, that is, as the EPU index increases (on a logarithmic scale), the unemployment rate decreases slightly. However, this effect is not statistically significant (p-value = 0.994), meaning we do not have enough evidence that changes in EPU have a material impact on the unemployment rate.
#The R-squared value is close to 0 (0.000),
# and the adjusted R-squared is also 0,
# indicating that the model barely explains any variability in the variables.