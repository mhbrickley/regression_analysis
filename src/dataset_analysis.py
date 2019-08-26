'''
Michael Brickley
Class: CS677 - Summer 2
Date: 8/22/2019
Description: Predict red wine quality on a scale of 0 - 10
'''

from warnings import simplefilter
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

### ADJUST SETTINGS###
# ignore SettingWithCopy warning
pd.options.mode.chained_assignment = None
# ignore FutureWarning
simplefilter(action='ignore', category=FutureWarning)
# print all columns of dataframe 
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 1000)

###IMPORT DATA###
try:
    # import csv data
    df = pd.read_csv('../winequality-red.csv')
except(FileNotFoundError):
    print("CSV file not found. Please review file path")

print('\nDataset preview below...')    
print(df.head()) 

###EXPLORATORY ANALYSIS OF TARGET FEATURE###
print("\nAverage quality rating: " + str(df['quality'].mean()))
print("Standard deviation: " + str(df['quality'].std()))
print("Max quality rating: " + str(df['quality'].max()))
print("Min quality rating: " + str(df['quality'].min()))
print("\nBreakdown of percentages of quality values:")
print(df['quality'].value_counts(normalize=True) * 100)

###CORRELATION###
# determine correlation between feature columns and target variable
corr = df.corr()['quality']

# remove quality column
corr = corr.drop('quality')

# print correlation of feature columns
print('\nPrinting correlation of feature columns...')
print(corr)

# plot results
plt.figure(0)
heatmap = sns.heatmap(df.corr(),cmap='coolwarm',annot=True) # annot=True to show correlation values in plot
fig = heatmap.get_figure()
fig.set_size_inches(10,10)
fig.savefig('../features_heatmap.png')
print('\nHeatmap of correlation values saved to root folder:  ../features_heatmap.png')

###DETERMINE NUMBER OF FEATURES TO USE###
# find absolute values of correlation values
corr = abs(corr)
# 0.20 for small to medium strength correlation
featureColumns = corr[(corr >= 0.20)].index.values.tolist()
print('\nFeature columns: ' + str(featureColumns))

# create training and testing data
x = df[featureColumns]
y = df['quality']

x_train,x_test,y_train,y_test = train_test_split(x, y, train_size=0.75, test_size= 0.25, random_state=1) # random_state should be anything but blank for consistent output

# create linear regression object
linReg = LinearRegression()
linReg.fit(x_train,y_train)

# predict quality values
predictions = linReg.predict(x_test) 
print('\nPrinting predictions...')
print(np.round(predictions))



###REVIEW MODEL METRICS###
print('\nPrinting model metrics...')
# accuracy
print('Linear regression model score: ' + str(linReg.score(x_test,y_test)))

# mean absolute error
mae = metrics.mean_absolute_error(y_test,predictions)
print('Mean absolute rrror: ' + str(mae))

# mean squared error
mse = metrics.mean_squared_error(y_test,predictions)
print('Mean squared error: ' + str(mse))

# root mean squared error (rmse)
# predict with training data for rmse calculation
trainedPredictions = linReg.predict(x_train)

rmseTrain = np.sqrt(metrics.mean_squared_error(y_train,trainedPredictions))
rmseTest = np.sqrt(metrics.mean_squared_error(y_test,predictions))
print('Root mean squared error of training data: ' + str(rmseTrain))
print('Root mean squared error of test data: ' + str(rmseTest))



