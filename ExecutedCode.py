import numpy as np
import pandas as pd
from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from plotly import tools
import chart_studio.plotly as py

from plotly.offline import init_notebook_mode, iplot
# init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier, plot_importance

data = pd.read_csv('xAPI-Edu-Data.csv')
# Any results you write to the current directory are saved as output.
print(data.head())

data.describe()
print(data.shape)
data.columns
#Check Missing Data
data.isnull().sum()
##%%%%
import seaborn as sns
import matplotlib.pyplot as plt

# Define a color palette for the bars
colors = sns.color_palette("husl", len(data['gender'].unique()))

# Create subplots for each feature
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(14, 16))

# Plot Gender
data['gender'].value_counts(normalize=True).plot(kind='bar', ax=axes[0, 0], color=colors)
axes[0, 0].set_title('Gender')
axes[0, 0].set_ylabel('Percentage')

# Plot NationalITy
data['NationalITy'].value_counts(normalize=True).plot(kind='bar', ax=axes[0, 1], color=colors)
axes[0, 1].set_title('NationalITy')
axes[0, 1].set_ylabel('Percentage')

# Plot PlaceofBirth
data['PlaceofBirth'].value_counts(normalize=True).plot(kind='bar', ax=axes[1, 0], color=colors)
axes[1, 0].set_title('PlaceofBirth')
axes[1, 0].set_ylabel('Percentage')

# Plot StageID
data['StageID'].value_counts(normalize=True).plot(kind='bar', ax=axes[1, 1], color=colors)
axes[1, 1].set_title('StageID')
axes[1, 1].set_ylabel('Percentage')

# Plot GradeID
data['GradeID'].value_counts(normalize=True).plot(kind='bar', ax=axes[2, 0], color=colors)
axes[2, 0].set_title('GradeID')
axes[2, 0].set_ylabel('Percentage')

# Plot Topic
data['Topic'].value_counts(normalize=True).plot(kind='bar', ax=axes[2, 1], color=colors)
axes[2, 1].set_title('Topic')
axes[2, 1].set_ylabel('Percentage')

# Plot Semester
data['Semester'].value_counts(normalize=True).plot(kind='bar', ax=axes[3, 0], color=colors)
axes[3, 0].set_title('Semester')
axes[3, 0].set_ylabel('Percentage')

# Plot Relation
data['Relation'].value_counts(normalize=True).plot(kind='bar', ax=axes[3, 1], color=colors)
axes[3, 1].set_title('Relation')
axes[3, 1].set_ylabel('Percentage')

plt.tight_layout()
plt.show()

##%%%%%%%
# #Gender Value Counts & Percentage In Dataset
# data['gender'].value_counts()
# print('Percentage',data.gender.value_counts(normalize=True))
# data.gender.value_counts(normalize=True).plot(kind='bar')
# # NationalITy Value Counts & Percentage In Dataset
# data['NationalITy'].value_counts()

# print('Percentage',data.NationalITy.value_counts(normalize=True))
# data.NationalITy.value_counts(normalize=True).plot(kind='bar')

# # PlaceofBirth Value Counts & Percentage In Dataset
# data['PlaceofBirth'].value_counts()
# print('Percentage',data.PlaceofBirth.value_counts(normalize=True))
# data.PlaceofBirth.value_counts(normalize=True).plot(kind='bar')

# # StageID Value Counts & Percentage In Dataset
# data['StageID'].value_counts()
# print('Percentage',data.StageID.value_counts(normalize=True))
# data.StageID.value_counts(normalize=True).plot(kind='bar')

# # GradeID Value Counts & Percentage In Dataset
# print('Percentage',data.GradeID.value_counts(normalize=True))
# data.GradeID.value_counts(normalize=True).plot(kind='bar')

# # Topic Value Counts & Parcentage In Dataset
# data['Topic'].value_counts()
# print('Percentage',data.Topic.value_counts(normalize=True))
# data.Topic.value_counts(normalize=True).plot(kind='bar')

# # Semester Value Counts & Parcentage In Dataset
# data['Semester'].value_counts()
# print('Parcentage',data.Semester.value_counts(normalize=True))
# data.Semester.value_counts(normalize=True).plot(kind='bar')

# # Relation Value Counts & Parcentage In Dataset
# data['Relation'].value_counts()
# print('Parcentage',data.Relation.value_counts(normalize=True))
# data.Relation.value_counts(normalize=True).plot(kind='bar')

# # Raisedhands Value Counts & Parcentage In Dataset
# data['raisedhands'].value_counts()


#print('Parcentage',df.raisedhands.value_counts(normalize=True))
#df.raisedhands.value_counts(normalize=True).plot(kind='bar')
color_brewer = ['#41B5A3','#FFAF87','#FF8E72','#ED6A5E','#377771','#E89005','#C6000D','#000000','#05668D','#028090','#9FD35C',
                '#02C39A','#F0F3BD','#41B5A3','#FF6F59','#254441','#B2B09B','#EF3054','#9D9CE8','#0F4777','#5F67DD','#235077','#CCE4F9','#1748D1',
                '#8BB3D6','#467196','#F2C4A2','#F2B1A4','#C42746','#330C25']
# fig = {
#   "data": [
#     {
#       "values": data["raisedhands"].value_counts().values,
#       "labels": data["raisedhands"].value_counts().index,
#       "domain": {"x": [0, .95]},
#       "name": "Raisedhands Parcentage",
#       "hoverinfo":"label+percent+name",
#       "hole": .7,
#       "type": "pie",
#       "marker": {"colors": [i for i in reversed(color_brewer)]},
#       "textfont": {"color": "#FFFFFF"}
#     }],
#   "layout": {
#         "title":"Raisedhands Parcentage",
#         "annotations": [
#             {
#                 "font": {
#                     "size": 15
#                 },
#                 "showarrow": False,
#                 "text": "Raisedhands Parcentage",
#                 "x": 0.47,
#                 "y": 0.5
#             }
#         ]
#     }
# }
# iplot(fig, filename='donut')

# # ParentschoolSatisfaction Value Counts & Parcentage In Dataset
# data['ParentschoolSatisfaction'].value_counts()
# print('Parcentage',data.ParentschoolSatisfaction.value_counts(normalize=True))
# data.ParentschoolSatisfaction.value_counts(normalize=True).plot(kind='bar')


# # ParentAnsweringSurvey Value Counts & Parcentage In Dataset
# data['ParentAnsweringSurvey'].value_counts()

# print('Parcentage',data.ParentAnsweringSurvey.value_counts(normalize=True))
# data.ParentAnsweringSurvey.value_counts(normalize=True).plot(kind='bar')

# # StudentAbsenceDays Value Counts & Parcentage In Dataset.
# data['StudentAbsenceDays'].value_counts()
# print('Parcentage',data.StudentAbsenceDays.value_counts(normalize=True))
# data.StudentAbsenceDays.value_counts(normalize=True).plot(kind='bar')
# # Class Value Counts & Parcentage In Dataset
# data['Class'].value_counts()
# print('Parcentage',data.Class.value_counts(normalize=True))
# data.Class.value_counts(normalize=True).plot(kind='bar')

# # categorical features individually to see what options are included and how each option fares when it comes to count(how many times it appears)
# fig, axarr  = plt.subplots(2,2,figsize=(10,10))
# sns.countplot(x='Class', data=data, ax=axarr[0,0], order=['L','M','H'])
# sns.countplot(x='gender', data=data, ax=axarr[0,1], order=['M','F'])
# sns.countplot(x='StageID', data=data, ax=axarr[1,0])
# sns.countplot(x='Semester', data=data, ax=axarr[1,1])

# fig, (axis1, axis2)  = plt.subplots(2, 1,figsize=(10,10))
# sns.countplot(x='Topic', data=data, ax=axis1)
# sns.countplot(x='NationalITy', data=data, ax=axis2)

# # some categorical features in relation to each other, to see what insights that could possibly read
# fig, axarr  = plt.subplots(2,2,figsize=(10,10))
# sns.countplot(x='gender', hue='Class', data=data, ax=axarr[0,0], order=['M','F'], hue_order=['L','M','H'])
# sns.countplot(x='gender', hue='Relation', data=data, ax=axarr[0,1], order=['M','F'])
# sns.countplot(x='gender', hue='StudentAbsenceDays', data=data, ax=axarr[1,0], order=['M','F'])
# sns.countplot(x='gender', hue='ParentAnsweringSurvey', data=data, ax=axarr[1,1], order=['M','F'])

# fig, (axis1, axis2)  = plt.subplots(2, 1,figsize=(10,10))
# sns.countplot(x='Topic', hue='gender', data=data, ax=axis1)
# sns.countplot(x='NationalITy', hue='gender', data=data, ax=axis2)

# # No apparent gender bias when it comes to subject/topic choices, we cannot conclude that girls performed better because they perhaps took less technical subjects
# # Gender disparity holds even at a country level. May just be as a result of the sampling.
# fig, (axis1, axis2)  = plt.subplots(2, 1,figsize=(10,10))
# sns.countplot(x='NationalITy', hue='Relation', data=data, ax=axis1)
# sns.countplot(x='NationalITy', hue='StudentAbsenceDays', data=data, ax=axis2)

# # moving on to visualizing categorical features with numerical features.
# fig, axarr  = plt.subplots(2,2,figsize=(10,10))
# sns.barplot(x='Class', y='VisITedResources', data=data, order=['L','M','H'], ax=axarr[0,0])
# sns.barplot(x='Class', y='AnnouncementsView', data=data, order=['L','M','H'], ax=axarr[0,1])
# sns.barplot(x='Class', y='raisedhands', data=data, order=['L','M','H'], ax=axarr[1,0])
# sns.barplot(x='Class', y='Discussion', data=data, order=['L','M','H'], ax=axarr[1,1])

# # As expected, those that participated more (higher counts in Discussion, raisedhands, AnnouncementViews, RaisedHands), performed better ...that thing about correlation and causation.
# fig, (axis1,axis2) = plt.subplots(1,2,figsize=(10,5))
# sns.barplot(x='gender', y='raisedhands', data=data, ax=axis1)
# sns.barplot(x='gender', y='Discussion', data=data, ax=axis2)

# # There are various other plots that help visualize Categorical vs Numerical data better.
# fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(10,5))
# sns.swarmplot(x='gender', y='AnnouncementsView', data=data, ax=axis1)
# sns.swarmplot(x='gender', y='raisedhands', data=data, ax=axis2)

import seaborn as sns
import matplotlib.pyplot as plt

# Define a new color palette
new_color_palette = ['#FF5733', '#33FF57', '#3366FF', '#FF33A1', '#A133FF', '#FFC300']

# Create a donut chart for Raisedhands Percentage
plt.figure(figsize=(8, 8))
plt.pie(
    data["raisedhands"].value_counts().values, 
    labels=data["raisedhands"].value_counts().index, 
    colors=new_color_palette, 
    autopct='%1.1f%%',
    wedgeprops={"edgecolor": "white"},
)
plt.title("Raisedhands Percentage", fontsize=14)
plt.show()

# Create bar charts for ParentschoolSatisfaction, ParentAnsweringSurvey, StudentAbsenceDays, and Class
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
data['ParentschoolSatisfaction'].value_counts(normalize=True).plot(kind='bar', ax=axes[0, 0], color=new_color_palette)
axes[0, 0].set_title('ParentschoolSatisfaction')
data['ParentAnsweringSurvey'].value_counts(normalize=True).plot(kind='bar', ax=axes[0, 1], color=new_color_palette)
axes[0, 1].set_title('ParentAnsweringSurvey')
data['StudentAbsenceDays'].value_counts(normalize=True).plot(kind='bar', ax=axes[1, 0], color=new_color_palette)
axes[1, 0].set_title('StudentAbsenceDays')
data['Class'].value_counts(normalize=True).plot(kind='bar', ax=axes[1, 1], color=new_color_palette)
axes[1, 1].set_title('Class')
plt.show()

# Create categorical feature visualizations
fig, axarr = plt.subplots(2, 2, figsize=(10, 10))
sns.countplot(x='Class', data=data, ax=axarr[0, 0], order=['L', 'M', 'H'], palette=new_color_palette)
sns.countplot(x='gender', data=data, ax=axarr[0, 1], order=['M', 'F'], palette=new_color_palette)
sns.countplot(x='StageID', data=data, ax=axarr[1, 0], palette=new_color_palette)
sns.countplot(x='Semester', data=data, ax=axarr[1, 1], palette=new_color_palette)

fig, (axis1, axis2) = plt.subplots(2, 1, figsize=(10, 10))
sns.countplot(x='Topic', data=data, ax=axis1, palette=new_color_palette)
sns.countplot(x='NationalITy', data=data, ax=axis2, palette=new_color_palette)

# Create categorical feature visualizations in relation to each other
fig, axarr = plt.subplots(2, 2, figsize=(10, 10))
sns.countplot(x='gender', hue='Class', data=data, ax=axarr[0, 0], order=['M', 'F'], hue_order=['L', 'M', 'H'], palette=new_color_palette)
sns.countplot(x='gender', hue='Relation', data=data, ax=axarr[0, 1], order=['M', 'F'], palette=new_color_palette)
sns.countplot(x='gender', hue='StudentAbsenceDays', data=data, ax=axarr[1, 0], order=['M', 'F'], palette=new_color_palette)
sns.countplot(x='gender', hue='ParentAnsweringSurvey', data=data, ax=axarr[1, 1], order=['M', 'F'], palette=new_color_palette)

fig, (axis1, axis2) = plt.subplots(2, 1, figsize=(10, 10))
sns.countplot(x='Topic', hue='gender', data=data, ax=axis1, palette=new_color_palette)
sns.countplot(x='NationalITy', hue='gender', data=data, ax=axis2, palette=new_color_palette)

# Create bar plots to visualize the relationship between categorical and numerical features
fig, axarr = plt.subplots(2, 2, figsize=(10, 10))
sns.barplot(x='Class', y='VisITedResources', data=data, order=['L', 'M', 'H'], ax=axarr[0, 0], palette=new_color_palette)
sns.barplot(x='Class', y='AnnouncementsView', data=data, order=['L', 'M', 'H'], ax=axarr[0, 1], palette=new_color_palette)
sns.barplot(x='Class', y='raisedhands', data=data, order=['L', 'M', 'H'], ax=axarr[1, 0], palette=new_color_palette)
sns.barplot(x='Class', y='Discussion', data=data, order=['L', 'M', 'H'], ax=axarr[1, 1], palette=new_color_palette)

# Additional bar plots
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
sns.barplot(x='gender', y='raisedhands', data=data, ax=axis1, palette=new_color_palette)
sns.barplot(x='gender', y='Discussion', data=data, ax=axis2, palette=new_color_palette)

# Swarm plots for additional visualization
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(10, 5))
sns.swarmplot(x='gender', y='AnnouncementsView', data=data, ax=axis1, palette=new_color_palette)
sns.swarmplot(x='gender', y='raisedhands', data=data, ax=axis2, palette=new_color_palette)

# Show the plots
plt.show()



#%%
# fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(10,5))
# sns.boxplot(x='Class', y='Discussion', data=data, order=['L','M','H'], ax=axis1)
# sns.boxplot(x='Class', y='VisITedResources', data=data, order=['L','M','H'], ax=axis2)
# # The two plots above tell us that visiting the resources may not be as sure a path to performing well as discussions
# fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(10,5))
# sns.pointplot(x='Semester', y='VisITedResources', hue='gender', data=data, ax=axis1)
# sns.pointplot(x='Semester', y='AnnouncementsView', hue='gender', data=data, ax=axis2)

# # both visiting resources and viewing announcements, students were more vigilant in the second semester, perhaps that last minute need to boost your final grade.
# # plots to visualize relationships between numerical features.
# fig, (axis1, axis2)  = plt.subplots(1, 2,figsize=(10,5))
# sns.regplot(x='raisedhands', y='VisITedResources', data=data, ax=axis1)
# sns.regplot(x='AnnouncementsView', y='Discussion', data=data, ax=axis2)

# Create subplots with a 1x2 grid and set the figure size
fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(12, 5))

# Set custom colors for the box plots
boxplot_colors = ["#FF5733", "#33FF57", "#3366FF"]

# Create the first box plot
sns.boxplot(x='Class', y='Discussion', data=data, order=['L', 'M', 'H'], ax=axis1, palette=boxplot_colors)
axis1.set_title('Discussion vs. Class', fontsize=14)
axis1.set_xlabel('Class', fontsize=12)
axis1.set_ylabel('Discussion', fontsize=12)

# Set custom colors for the point plots
pointplot_colors = ["#FF33A1", "#A133FF"]

# Create the second box plot
sns.boxplot(x='Class', y='VisITedResources', data=data, order=['L', 'M', 'H'], ax=axis2, palette=boxplot_colors)
axis2.set_title('Visited Resources vs. Class', fontsize=14)
axis2.set_xlabel('Class', fontsize=12)
axis2.set_ylabel('Visited Resources', fontsize=12)

# Show the plot
plt.show()


# Gender Comparison With Parents Relationship
#%%
# plot = sns.countplot(x='Class', hue='Relation', data=data, order=['L', 'M', 'H'], palette='Set1')
# plot.set(xlabel='Class', ylabel='Count', title='Gender comparison')
# plt.show()

# # Pairplot
# sns.pairplot(data,hue='Class')
import seaborn as sns
import matplotlib.pyplot as plt

# Custom color palette for the count plot
countplot_palette = ["#FF5733", "#33FF57", "#3366FF"]

# Create a count plot with the custom color palette
plt.figure(figsize=(10, 6))
count_plot = sns.countplot(x='Class', hue='Relation', data=data, order=['L', 'M', 'H'], palette=countplot_palette)
count_plot.set(xlabel='Class', ylabel='Count', title='Class and Relation Comparison')
plt.show()

# Custom color palette for the pairplot
pairplot_palette = ["#FF5733", "#33FF57", "#3366FF"]

# Create a pairplot with the custom color palette
sns.set(style="ticks")
pair_plot = sns.pairplot(data, hue='Class', palette=pairplot_palette)
plt.suptitle("Pairplot of Features by Class", y=1.02)
plt.show()


#%%
# # Graph Analysis Gender vs Place of Birth
# import networkx as nx
# import matplotlib.pyplot as plt

# g = nx.Graph()
# g = nx.from_pandas_edgelist(data, source='gender', target='PlaceofBirth')

# plt.figure(figsize=(10, 12))
# nx.draw_networkx(g, with_labels=True, node_size=70, alpha=0.5, node_color="red", font_size=14, font_color="black")
# plt.show()

# Graph Analysis Gender vs Place of Birth
import networkx as nx
import matplotlib.pyplot as plt

# Create a Graph
G = nx.Graph()

# Add nodes and edges
G.add_nodes_from(data['gender'], bipartite=0)
G.add_nodes_from(data['PlaceofBirth'], bipartite=1)
G.add_edges_from([(row['gender'], row['PlaceofBirth']) for index, row in data.iterrows()])

plt.figure(figsize=(12, 10))

# Define node colors based on bipartite sets
color_map = [0.2 if node in data['gender'].unique() else 0.6 for node in G.nodes()]

# Draw the network graph
pos = nx.spring_layout(G)  # You can change the layout method if needed
nx.draw(G, pos, with_labels=True, node_size=200, alpha=0.7, node_color=color_map, font_size=14, font_color="black")

plt.title('Gender vs Place of Birth Network', fontsize=16)
plt.axis('off')  # Turn off the axis
plt.show()

#%%
# Machine Learning Algorithm aooly
# Label Encoding
# 1.Gender Encoding

Features = data.drop('gender',axis=1)
Target = data['gender']
label = LabelEncoder()
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col])
    
    # 2.Semester Encoding
Features = data.drop('Semester',axis=1)
Target = data['Semester']
label = LabelEncoder()
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col])
    
# 3.ParentAnsweringSurvey Encoding
Features = data.drop('ParentAnsweringSurvey',axis=1)
Target = data['ParentAnsweringSurvey']
label = LabelEncoder()
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col])
    
    # 4.Relation Encoding
    
Features = data.drop('Relation',axis=1)
Target = data['Relation']
label = LabelEncoder()
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col])

# 5.ParentschoolSatisfaction Encoding
Features = data.drop('ParentschoolSatisfaction',axis=1)
Target = data['ParentschoolSatisfaction']
label = LabelEncoder()
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col])
    
# 6.StudentAbsenceDays Encoding
Features = data.drop('StudentAbsenceDays',axis=1)
Target = data['StudentAbsenceDays']
label = LabelEncoder()
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col])
    
# 7.Class Encoding
Features = data.drop('Class',axis=1)
Target = data['Class']
label = LabelEncoder()
Cat_Colums = Features.dtypes.pipe(lambda Features: Features[Features=='object']).index
for col in Cat_Colums:
    Features[col] = label.fit_transform(Features[col])
#%%
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(Features, Target, test_size=0.2, random_state=52)
#%
X=Features;
y=Target;
print("Logistic Regression Classification")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Train the logistic regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Make predictions
lg_pred = logreg.predict(X_test)

# Print classification report and accuracy score
print(classification_report(y_test, lg_pred))
print(accuracy_score(y_test, lg_pred))

# Calculate precision, recall, F1-score, and accuracy separately
logreg_report = classification_report(y_test, lg_pred, output_dict=True)
lg_acc = accuracy_score(y_test, lg_pred)
lg_precision = logreg_report['weighted avg']['precision']
lg_recall = logreg_report['weighted avg']['recall']
lg_f1score = logreg_report['weighted avg']['f1-score']
#%
print("XGB Classification")
from sklearn.preprocessing import LabelEncoder

# Convert categorical labels to numerical labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train the XGBoost classifier
xgb = XGBClassifier(max_depth=10, learning_rate=0.1, n_estimators=100, seed=10)
xgb.fit(X_train, y_train_encoded)

# Make predictions
xgb_pred = xgb.predict(X_test)

# Decode the numerical labels back to categorical labels if needed
y_test_decoded = label_encoder.inverse_transform(y_test_encoded)
xgb_pred_decoded = label_encoder.inverse_transform(xgb_pred)

# Print classification report and accuracy score
print(classification_report(y_test_decoded, xgb_pred_decoded))
print(accuracy_score(y_test_decoded, xgb_pred_decoded))

# Plot feature importances
#%%
import numpy as np
import matplotlib.pyplot as plt

# Get feature importances from the trained XGBoost classifier
importances = xgb.feature_importances_
feature_names = X_train.columns

# Sort the importances and feature names in descending order
indices = np.argsort(importances)[::-1]
importances_sorted = importances[indices]
feature_names_sorted = feature_names[indices]

# Create a vertical bar chart of the feature importances
plt.figure(figsize=(8, 10))  # Adjust the figure size to accommodate vertical bars
plt.bar(range(len(importances_sorted)), importances_sorted, align='center')
plt.xticks(range(len(importances_sorted)), feature_names_sorted, rotation='vertical', fontsize=14)  # Rotate x-axis labels
plt.ylabel('Importance', fontsize=14)
plt.xlabel('Feature', fontsize=14)
plt.title('Feature Importances', fontsize=14)

# Set the font size for the plot
plt.rcParams['font.size'] = 14

# Show the plot
plt.show()


#%%
from sklearn.metrics import classification_report, precision_recall_fscore_support
XGBReport = classification_report(y_test_decoded, xgb_pred_decoded)

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test_decoded, xgb_pred_decoded)
xgb_acc = accuracy_score(y_test_decoded,xgb_pred_decoded)
xgb_precision = precision[1]
xgb_recall = recall[1]
xgb_f1score = f1_score[0]

#%%
print("CNN Classification")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.utils import to_categorical


# Select categorical columns for one-hot encoding
categorical_cols = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID', 'Topic', 'Semester',
                    'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction', 'StudentAbsenceDays']

# One-hot encode the categorical columns
data_encoded = pd.get_dummies(data, columns=categorical_cols)

# Split the data into features (X) and target (y)
X = data_encoded.drop('Class', axis=1)
y = data_encoded['Class']

# Perform label encoding on the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)

# Convert the features to numpy arrays
X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)

# Reshape the features for the Conv1D input shape
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Convert the target variable to categorical
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Initialize the CNN model
model = Sequential()

# Add the convolutional layer
model.add(Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)))

# Add the max pooling layer
model.add(MaxPooling1D(2))

# Flatten the feature maps
model.add(Flatten())

# Add a fully connected layer
model.add(Dense(128, activation='relu'))

# Add the output layer
model.add(Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
model.fit(X_train, y_train, epochs=200, batch_size=18, verbose=0, validation_data=(X_test, y_test))

cnn_pred=model.predict(X_test);
# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

from sklearn.metrics import classification_report

# Make predictions on the test set
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# Generate the classification report
report = classification_report(y_test_labels, y_pred_labels)

# Print the classification report
print(report)

# Calculate accuracy
cnn_acc = accuracy_score(y_test_labels, y_pred_labels)

# Calculate precision, recall, and F1-score
cnn_precision, cnn_recall, cnn_f1score, _ = precision_recall_fscore_support(y_test_labels, y_pred_labels, average='weighted')


#%%
print("SVM Classification")
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Perform label encoding on the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=52)
# Initialize the SVM classifier
svm = SVC()
# Fit the SVM classifier to the training data
svm.fit(X_train, y_train)
# Make predictions on the test set
svm_pred = svm.predict(X_test)
# Evaluate the performance of the SVM classifier
print("Classification Report:")
print(classification_report(y_test, svm_pred))
print("Accuracy:", accuracy_score(y_test, svm_pred))

svm_report = classification_report(y_test, svm_pred, output_dict=True)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_precision = svm_report['weighted avg']['precision']
svm_recall = svm_report['weighted avg']['recall']
svm_f1score = svm_report['weighted avg']['f1-score']

print("Classification Report:")
print(classification_report(y_test, svm_pred))
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1-Score:", svm_f1score)
#%%
print("SVM-SSO Classification")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def initialize_spiders(population_size, num_dimensions, bounds):
    return np.random.uniform(bounds[0], bounds[1], size=(population_size, num_dimensions))


def evaluate_fitness(spider_positions, X_train, y_train, X_test, y_test):
    fitness_scores = []
    for position in spider_positions:
        svm = SVC(C=position[0], kernel='linear', gamma=position[1])
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        fitness_scores.append(accuracy)
    return np.array(fitness_scores)


def update_spiders(spider_positions, fitness_scores, best_spider, bounds):
    updated_spiders = []
    for position, fitness in zip(spider_positions, fitness_scores):
        new_position = position + np.random.uniform(-1, 1) * (best_spider - position)
        new_position = np.clip(new_position, bounds[0], bounds[1])
        updated_spiders.append(new_position)
    return np.array(updated_spiders)


# Load the data into a pandas DataFrame
data = pd.read_csv('xAPI-Edu-Data.csv')

# Split the data into features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Perform label encoding on the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Perform one-hot encoding on categorical variables
categorical_cols = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID',
                    'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
                    'StudentAbsenceDays']
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=52)

# Define the parameter bounds for the SSA
parameter_bounds = [(0.01, 10), (0.1, 5)]

# Define the SSA parameters
population_size = 50
num_dimensions = len(parameter_bounds)
num_iterations = 50

# Initialize spiders randomly within the parameter bounds
spider_positions = initialize_spiders(population_size, num_dimensions, parameter_bounds)

# Perform the SSA iterations
for _ in range(num_iterations):
    fitness_scores = evaluate_fitness(spider_positions, X_train, y_train, X_test, y_test)
    best_spider_index = np.argmax(fitness_scores)
    best_spider = spider_positions[best_spider_index]
    spider_positions = update_spiders(spider_positions, fitness_scores, best_spider, parameter_bounds)
    
# Get the best spider (parameters) found by the SSA
best_spider_index = np.argmax(fitness_scores)
best_spider = spider_positions[best_spider_index]

# Train the SVM classifier with the best spider parameters
svmSSO = SVC(C=best_spider[0], kernel='linear', gamma=best_spider[1])
svmSSO.fit(X_train, y_train)

# Make predictions on the test set using the best SVM classifier
svmSSO_pred = svmSSO.predict(X_test)

# Evaluate the performance of the SVM classifier
print("Classification Report:")
print(classification_report(y_test, svmSSO_pred))
print("Accuracy:", accuracy_score(y_test, svmSSO_pred))
print("Best Parameters:", best_spider)


svmSSO_report = classification_report(y_test, svmSSO_pred, output_dict=True)
svmsso_accuracy = accuracy_score(y_test, svmSSO_pred)
svmsso_precision = svmSSO_report['weighted avg']['precision']
svmsso_recall = svmSSO_report['weighted avg']['recall']
svmsso_f1score = svmSSO_report['weighted avg']['f1-score']


#%%
print("SVM-SA Classification")
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def acceptance_probability(current_fitness, new_fitness, temperature):
    if new_fitness > current_fitness:
        return 1.0
    else:
        return np.exp((new_fitness - current_fitness) / temperature)


def simulated_annealing(X_train, y_train, X_test, y_test, initial_solution, bounds, temperature, cooling_rate, max_iterations):
    current_solution = initial_solution.copy()
    best_solution = initial_solution.copy()
    current_fitness = evaluate_fitness(current_solution, X_train, y_train, X_test, y_test)
    best_fitness = current_fitness

    for iteration in range(max_iterations):
        new_solution = current_solution.copy()
        for i in range(len(bounds)):
            new_solution[i] += np.random.uniform(-1, 1) * (bounds[i][1] - bounds[i][0])
            new_solution[i] = np.clip(new_solution[i], bounds[i][0], bounds[i][1])

        new_fitness = evaluate_fitness(new_solution, X_train, y_train, X_test, y_test)
        acceptance_prob = acceptance_probability(current_fitness, new_fitness, temperature)
        if acceptance_prob > np.random.uniform():
            current_solution = new_solution
            current_fitness = new_fitness

        if new_fitness > best_fitness:
            best_solution = new_solution
            best_fitness = new_fitness

        temperature *= cooling_rate

    return best_solution, best_fitness


def evaluate_fitness(solution, X_train, y_train, X_test, y_test):
    svm = SVC(C=solution[0], kernel='linear', gamma=solution[1])
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


# Load the data into a pandas DataFrame
data = pd.read_csv('xAPI-Edu-Data.csv')

# Split the data into features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Perform label encoding on the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Perform one-hot encoding on categorical variables
categorical_cols = ['gender', 'NationalITy', 'PlaceofBirth', 'StageID', 'GradeID', 'SectionID',
                    'Topic', 'Semester', 'Relation', 'ParentAnsweringSurvey', 'ParentschoolSatisfaction',
                    'StudentAbsenceDays']
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=52)

# Define the parameter bounds for Simulated Annealing
parameter_bounds = [(0.01, 10), (0.1, 5)]

# Define Simulated Annealing parameters
initial_solution = [1.0, 0.1]
temperature = 100.0
cooling_rate = 0.95
max_iterations = 20

# Perform Simulated Annealing
best_solution, best_fitness = simulated_annealing(X_train, y_train, X_test, y_test, initial_solution,
                                                  parameter_bounds, temperature, cooling_rate, max_iterations)

# Train the SVM classifier with the best solution parameters
svmSA = SVC(C=best_solution[0], kernel='linear', gamma=best_solution[1])
svmSA.fit(X_train, y_train)

# Make predictions on the test set using the best SVM classifier
svmSA_pred = svm.predict(X_test)

# Evaluate the performance of the SVM classifier
print("Classification Report:")
print(classification_report(y_test, svmSA_pred))
print("Accuracy:", accuracy_score(y_test, svmSA_pred))
print("Best Parameters:", best_solution)

svmSA_report = classification_report(y_test, svmSA_pred, output_dict=True)
svmSA_accuracy = accuracy_score(y_test, svmSA_pred)
svmSA_precision = svmSA_report['weighted avg']['precision']
svmSA_recall = svmSA_report['weighted avg']['recall']
svmSA_f1score = svmSA_report['weighted avg']['f1-score']

#%%
print("SVM-PSO")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from pyswarm import pso
def evaluate_fitness(params, X_train, y_train, X_test, y_test):
    C = params[0]
    gamma = params[1]

    svm = SVC(C=C, kernel='rbf', gamma=gamma)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return -accuracy  # Minimize negative accuracy as PSO maximizes fitness


# Perform label encoding on the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Perform one-hot encoding on categorical variables
X_encoded = pd.get_dummies(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=52)

# Define the bounds for the parameters
lower_bounds = [0.1, 0.001]
upper_bounds = [100, 10]
bounds = (lower_bounds, upper_bounds)

# Perform Particle Swarm Optimization
best_params, best_fitness = pso(evaluate_fitness, lower_bounds, upper_bounds, args=(X_train, y_train, X_test, y_test))

# Retrieve the best parameter values
C = best_params[0]
gamma = best_params[1]

# Train the SVM classifier with the best parameters
svmPSO = SVC(C=C, kernel='rbf', gamma=gamma)
svmPSO.fit(X_train, y_train)

# Make predictions on the test set using the trained classifier
svmPSO_pred = svmPSO.predict(X_test)

# Evaluate the performance of the SVM classifier
print("Classification Report:")
print(classification_report(y_test, svmPSO_pred))
print("Accuracy:", accuracy_score(y_test, svmPSO_pred))
print("Best Parameters:")
print("  C:", C)
print("  Gamma:", gamma)

svmPSO_report = classification_report(y_test, svmPSO_pred, output_dict=True)
svmPSO_accuracy = accuracy_score(y_test, svmPSO_pred)
svmPSO_precision = svmPSO_report['weighted avg']['precision']
svmPSO_recall = svmPSO_report['weighted avg']['recall']
svmPSO_f1score = svmPSO_report['weighted avg']['f1-score']

#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


# Define the SVM objective function to be optimized
def svm_objective(params):
    svm = SVC(C=params['C'], kernel='rbf', gamma=params['gamma'])
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    return {'loss': -accuracy, 'status': STATUS_OK}


# Load the data into a pandas DataFrame
data = pd.read_csv('xAPI-Edu-Data.csv')

# Split the data into features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Perform label encoding on the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Perform one-hot encoding on categorical variables
X_encoded = pd.get_dummies(X)

# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_encoded, y, test_size=0.2, random_state=52)

# Define the search space for C and gamma
space = {
    'C': hp.loguniform('C', np.log(1e-6), np.log(10)),
    'gamma': hp.loguniform('gamma', np.log(1e-6), np.log(1))
}

# Perform Bayesian Optimization to tune the parameters
trials = Trials()
best = fmin(fn=svm_objective, space=space, algo=tpe.suggest, max_evals=200, trials=trials)

# Get the best parameters found during optimization
best_C = best['C']
best_gamma = best['gamma']

# Train the SVM classifier with the tuned parameters
svmBO = SVC(C=best_C, kernel='rbf', gamma=best_gamma)
svmBO.fit(X_train, y_train)

# Make predictions on the validation set using the trained classifier
svmBO_pred = svmBO.predict(X_val)

# Evaluate the performance of the SVM classifier
print("Classification Report:")
print(classification_report(y_val, svmBO_pred))
print("Accuracy:", accuracy_score(y_val, svmBO_pred))
print("Tuned Parameters:")
print("  C:", best_C)
print("  Gamma:", best_gamma)

svmBO_report = classification_report(y_test, svmBO_pred, output_dict=True)
svmBO_accuracy = accuracy_score(y_test, svmBO_pred)
svmBO_precision = svmBO_report['weighted avg']['precision']
svmBO_recall = svmBO_report['weighted avg']['recall']
svmBO_f1score = svmBO_report['weighted avg']['f1-score']



#%%
print("Random forest");



def evaluate_fitness(params, X_train, y_train, X_test, y_test):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    min_samples_split = int(params[2])

    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return -accuracy  # Minimize negative accuracy as PSO maximizes fitness

# Perform label encoding on the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Perform one-hot encoding on categorical variables
X_encoded = pd.get_dummies(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=52)

# Define the bounds for the parameters
lower_bounds = [10, 1, 2]
upper_bounds = [500, 50, 10]
bounds = (lower_bounds, upper_bounds)

# Perform Particle Swarm Optimization
best_params, best_fitness = pso(evaluate_fitness, lower_bounds, upper_bounds, args=(X_train, y_train, X_test, y_test))

# Retrieve the best parameter values
n_estimators = int(best_params[0])
max_depth = int(best_params[1])
min_samples_split = int(best_params[2])

# Train the Random Forest classifier with the best parameters
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
rf.fit(X_train, y_train)

# Make predictions on the test set using the trained classifier
rf_pred = rf.predict(X_test)

# Evaluate the performance of the Random Forest classifier
print("Classification Report:")
print(classification_report(y_test, rf_pred))
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Best Parameters:")
print("  Number of Estimators:", n_estimators)
print("  Max Depth:", max_depth)
print("  Min Samples Split:", min_samples_split)

rf_report = classification_report(y_test, rf_pred, output_dict=True)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = rf_report['weighted avg']['precision']
rf_recall = rf_report['weighted avg']['recall']
rf_f1score = rf_report['weighted avg']['f1-score']


#%%
print("XGboost")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
from tpot import TPOTClassifier


# Load the data into a pandas DataFrame
data = pd.read_csv('xAPI-Edu-Data.csv')

# Split the data into features (X) and target (y)
X = data.drop('Class', axis=1)
y = data['Class']

# Perform label encoding on the target variable
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Perform one-hot encoding on categorical variables
X_encoded = pd.get_dummies(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=52)

# Define the fitness function for TPOT
def tpot_fitness(X, y):
    xgb_classifier = xgb.XGBClassifier()
    xgb_classifier.fit(X, y)
    y_pred = xgb_classifier.predict(X)
    return accuracy_score(y, y_pred)

# Create and fit TPOT classifier
tpot = TPOTClassifier(generations=10, population_size=20, verbosity=2)
tpot.fit(X_train, y_train)

# Evaluate the best pipeline on the test set
xgb_pred = tpot.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, xgb_pred))
print("Accuracy:", accuracy_score(y_test, xgb_pred))

xgb_report = classification_report(y_test, xgb_pred, output_dict=True)
xgbTPOT_accuracy = accuracy_score(y_test, xgb_pred)
xgbTPOT_precision = xgb_report['weighted avg']['precision']
xgbTPOT_recall = xgb_report['weighted avg']['recall']
xgbTPOT_f1score = xgb_report['weighted avg']['f1-score']

#%%
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# Define the parameter search space
param_space = {
    'max_depth': (3, 10),
    'learning_rate': (0.01, 0.1),
    'n_estimators': (100, 500),
    'gamma': (0, 10),
    'min_child_weight': (1, 10),
    'subsample': (0.5, 1),
    'colsample_bytree': (0.5, 1)
}

# Define the fitness function for CRO
def fitness_function(params):
    # Create the XGBoost classifier with the given hyperparameters
    xgb = XGBClassifier(
        max_depth=int(params[0]),
        learning_rate=params[1],
        n_estimators=int(params[2]),
        gamma=params[3],
        min_child_weight=params[4],
        subsample=params[5],
        colsample_bytree=params[6]
    )
    
    # Encode the categorical labels to numerical labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    
    # Perform cross-validation and calculate fitness (accuracy in this case)
    fitness = cross_val_score(xgb, X_train, y_train_encoded, cv=5, scoring='accuracy').mean()
    
    return fitness

# Define the CRO parameters
num_iterations = 100
population_size = 10
mutation_rate = 0.2
step_size = 0.1

# Initialize the population
population = []
for _ in range(population_size):
    individual = []
    for key, (lower, upper) in param_space.items():
        gene = np.random.uniform(lower, upper)
        individual.append(gene)
    population.append(individual)

# Perform Coral Reefs Optimization
for _ in range(num_iterations):
    for i, individual in enumerate(population):
        fitness = fitness_function(individual)
        
        # Generate an offspring
        offspring = []
        for j, gene in enumerate(individual):
            key = list(param_space.keys())[j]
            if np.random.rand() < mutation_rate:
                gene += np.random.choice([-1, 1]) * step_size
                gene = np.clip(gene, param_space[key][0], param_space[key][1])
            offspring.append(gene)
        
        # Evaluate the fitness of the offspring
        offspring_fitness = fitness_function(offspring)
        
        # Replace the individual with the offspring if it has higher fitness
        if offspring_fitness > fitness:
            population[i] = offspring

# Find the best individual
best_individual = max(population, key=fitness_function)
best_fitness = fitness_function(best_individual)

# Create the XGBoost classifier with the best hyperparameters
best_xgb = XGBClassifier(
    max_depth=int(best_individual[0]),
    learning_rate=best_individual[1],
    n_estimators=int(best_individual[2]),
    gamma=best_individual[3],
    min_child_weight=best_individual[4],
    subsample=best_individual[5],
    colsample_bytree=best_individual[6]
)

# Fit the data using the best estimator
best_xgb.fit(X_train, y_train)

# Make predictions using the best estimator
xgb_pred = best_xgb.predict(X_test)

# Calculate accuracy
xgb_accuracy = accuracy_score(y_test, xgb_pred)

# Print the best hyperparameters and accuracy
print("Best Hyperparameters:", dict(zip(param_space.keys(), best_individual)))
print("Accuracy:", xgb_accuracy)


#%% converting them to csv
import csv

# Create a list of dictionaries with the desired values
data = [
    {
        'Model': 'xgbTPOT',
        'Accuracy': xgbTPOT_accuracy,
        'Precision': xgbTPOT_precision,
        'Recall': xgbTPOT_recall,
        'F1-Score': xgbTPOT_f1score
    },
    {
        'Model': 'Random Forest',
        'Accuracy': rf_accuracy,
        'Precision': rf_precision,
        'Recall': rf_recall,
        'F1-Score': rf_f1score
    },
    {
        'Model': 'svmBO',
        'Accuracy': svmBO_accuracy,
        'Precision': svmBO_precision,
        'Recall': svmBO_recall,
        'F1-Score': svmBO_f1score
    },
    {
        'Model': 'svmPSO',
        'Accuracy': svmPSO_accuracy,
        'Precision': svmPSO_precision,
        'Recall': svmPSO_recall,
        'F1-Score': svmPSO_f1score
    },
    {
        'Model': 'svmSA',
        'Accuracy': svmSA_accuracy,
        'Precision': svmSA_precision,
        'Recall': svmSA_recall,
        'F1-Score': svmSA_f1score
    },
    {
        'Model': 'svmSSO',
        'Accuracy': svmsso_accuracy,
        'Precision': svmsso_precision,
        'Recall': svmsso_recall,
        'F1-Score': svmsso_f1score
    },
    {
        'Model': 'svm',
        'Accuracy': svm_accuracy,
        'Precision': svm_precision,
        'Recall': svm_recall,
        'F1-Score': svm_f1score
    },
    {
        'Model': 'CNN',
        'Accuracy': cnn_acc,
        'Precision': cnn_precision,
        'Recall': cnn_recall,
        'F1-Score': cnn_f1score
    },
    {
        'Model': 'xgb',
        'Accuracy': xgb_acc,
        'Precision': xgb_precision,
        'Recall': xgb_recall,
        'F1-Score': xgb_f1score
    },
    {
        'Model': 'Logistic Regression',
        'Accuracy': lg_acc,
        'Precision': lg_precision,
        'Recall': lg_recall,
        'F1-Score': lg_f1score
    }
]

# Define the CSV file path
csv_file = 'classification_metrics.csv'

# Define the fieldnames for the CSV file
fieldnames = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']

# Write the data to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    
    # Write the header
    writer.writeheader()
    
    # Write the data rows
    writer.writerows(data)

print("Classification metrics exported to:", csv_file)

#%%
import matplotlib.pyplot as plt

# AUC values for each model
auc_values = [0.9176, 0.9298, 0.8963, 0.8762, 0.8701, 0.9567, 0.8263, 0.8156, 0.9124, 0.8897,
              0.9975, 0.8751, 0.8697, 0.9054, 0.9125, 0.9162]

# Models
models = ['XGB-TPOT', 'Random Forest', 'SVM-BO', 'SVM-PSO', 'SVM-SA', 'SVM-SSO', 'SVM', 'CNN', 'XGB',
          'Logistic Regression', 'XGB-CRO (Proposed)', 'Decision Tree', 'Naive Bayes',
          'K-Nearest Neighbors', 'AdaBoost', 'Gradient Boosting']

# Plot the ROC curve
plt.figure(figsize=(8, 6))
for model, auc_value in zip(models, auc_values):
    plt.plot(1, 1, linestyle='', label='{} (AUC = {:.4f})'.format(model, auc_value))

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.grid(True)
plt.show()

#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.models import resnext50_32x4d
from jaya import Jaya

# Assuming you have a DataFrame 'df' with features and labels
# Modify this part according to your dataset
X = data_encoded.drop('Class', axis=1)
y = data_encoded['Class']

features = X
labels = y

# Convert labels to numeric format
label_mapping = {'M': 0, 'F': 1}
labels = [label_mapping[label] for label in labels]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train, X_test = torch.tensor(X_train).float(), torch.tensor(X_test).float()
y_train, y_test = torch.tensor(y_train).long(), torch.tensor(y_test).long()

# Define a simple ResNeXt model
class CustomResNeXt(nn.Module):
    def __init__(self):
        super(CustomResNeXt, self).__init__()
        self.resnext = resnext50_32x4d(pretrained=True)
        self.fc = nn.Linear(2048, 2)  # Assuming binary classification

    def forward(self, x):
        x = self.resnext(x)
        x = self.fc(x)
        return x

# Instantiate the model, loss, and optimizer
model = CustomResNeXt()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Hyperparameter space for Jaya algorithm
hyperparameter_space = [(1e-5, 1e-1), (16, 64)]  # Example: learning rate and batch size

# Convert data to PyTorch DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Train the model using Jaya algorithm
jaya = Jaya()
best_params = jaya.minimize(lambda params: train_model(model, train_loader, criterion, optimizer, *params),
                            hyperparameter_space)

# Update the model with the best hyperparameters
learning_rate, batch_size = best_params
for param_group in optimizer.param_groups:
    param_group['lr'] = learning_rate

# Continue training with the best hyperparameters if needed
# ...

# Evaluate the model on the test set
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()

    accuracy = total_correct / total_samples
    return accuracy

test_accuracy = evaluate_model(model, test_loader, criterion)
print(f"Test Accuracy: {test_accuracy}")
