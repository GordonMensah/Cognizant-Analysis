#!/usr/bin/env python
# coding: utf-8

# # Social Buzz Analysis

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# ## Data Cleaning

# Wrangling Reactions

# In[2]:


R_df=pd.read_csv('reactions.csv')


# In[3]:


R_df


# In[4]:


R_df=R_df.drop(columns=['Unnamed: 0', 'User ID'])


# In[5]:


R_df= R_df.dropna()


# In[6]:


R_df


# In[7]:


R_df['Type'].unique()


# In[8]:


R_df['Type']=R_df['Type'].replace({'disgust':'Disgust', 'dislike':'Dislike', 'scared':'Scared',
                                  'interested':'Interested', 'peeking':'Peeking','cherish':'Cherish','hate':'Hate',
                                  'love':'Love', 'indifferent':'Indifferent', 'super love':'super Love', 'intrigued':'Intrigued',
                                  'worried':'Worried', 'like':'Like', 'heart':'Heart', 'want':'Want', 'adore':'Adore'})


# In[9]:


R_df['Datetime'] = pd.to_datetime(R_df['Datetime']).dt.strftime('%Y-%m-%d, %H-%M-%S')


# In[10]:


R_df = R_df.rename(columns={'Type': 'Reaction_Type'})


# In[11]:


R_df


# Exporting  reactions as Reactions_df

# In[12]:


Reactions_df=R_df.to_csv('Reactions_df.csv', index=False)


# 

# 

# In[ ]:





# Wrangling Reaction Types

# In[13]:


RT_df=pd.read_csv('ReactionTypes.csv')


# In[14]:


RT_df


# In[15]:


RT_df=RT_df.drop(columns=['Unnamed: 0'])


# In[16]:


RT_df['Type']=RT_df['Type'].replace({'disgust':'Disgust', 'dislike':'Dislike', 'scared':'Scared',
                                  'interested':'Interested', 'peeking':'Peeking','cherish':'Cherish','hate':'Hate',
                                  'love':'Love', 'indifferent':'Indifferent', 'super love':'super Love', 'intrigued':'Intrigued',
                                  'worried':'Worried', 'like':'Like', 'heart':'Heart', 'want':'Want', 'adore':'Adore'})


# In[17]:


RT_df['Sentiment']=RT_df['Sentiment'].replace({'positive':'Positive', 'negative':'Negative','neutral':'Neutral'})


# In[18]:


RT_df


# In[19]:


RT_df = RT_df.rename(columns={'Type': 'Reaction_Type'})


# Exporting Reaction types as Reaction_Tpes_df

# In[20]:


ReactionTypes_df=RT_df.to_csv('ReactionTypes_df.csv', index=False)


# In[ ]:





# In[ ]:





# Wrangling Content

# In[21]:


C_df=pd.read_csv("Content.csv")


# In[22]:


C_df


# In[23]:


C_df=C_df.drop(columns=['Unnamed: 0','User ID','URL'])


# In[24]:


C_df


# In[25]:


C_df.dropna()


# In[26]:


C_df['Type'].unique()


# In[27]:


C_df['Type']=C_df['Type'].replace({'photo':'Photo','video':'Video','audio':'Audio'})


# In[28]:


C_df['Category'].unique()


# In[29]:


C_df['Category']=C_df['Category'].replace({'healthy eating':'Healthy Eating', 'technology':'Technology', 'food':'Food', 'cooking':'Cooking',
                                           'dogs':'Dogs', 'soccer':'Soccer', 'public speaking':'Public Speaking', 'science': 'Science', 'tennis':'Tennis', 
                                           'travel': 'Travel','fitness':'Fitness', 'education':'Education', 'studying':'Studying', 'veganism':'Veganism',
                                           'animals':'Animals','culture':'Culture', '"culture"':'Culture', '"studying"':'Studying','"animals"':'Animals', '"soccer"':'Soccer',
                                           '"dogs"':'Dogs','"tennis"':'Tennis','"food"':'Food','"technology"':'Technology', '"cooking"':'Cooking','"public speaking"':'Public Speaking',
                                           '"veganism"':'Veganism','"science"':'Science'})


# In[30]:


C_df


# In[31]:


C_df=C_df.rename(columns={'Type':'Reaction_Type'})


# In[32]:


C_df


# Exporting content as Content_df

# In[33]:


Content_df=C_df.to_csv("Content_df.csv")


# ## Data Was Exported to SQL for further Table Aggregation

# In[ ]:





# In[34]:


import seaborn as nsns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Importing Merged Data

# In[35]:


SB=pd.read_csv("Merged.csv")


# In[36]:


SB.tail()


# ## Feature Transformation

# ## Which month has the most Traffic?

# In[37]:


SB['datetime'] = pd.to_datetime(SB['datetime'])

SB['month'] = SB['datetime'].dt.month


# In[38]:


month_counts=SB['month'].value_counts()


# In[39]:


month_counts


# In[40]:


plt.bar(month_counts.index, month_counts.values , color='purple')

plt.xlabel('Month')
plt.ylabel('Count')
plt.title('Count of each Month')
plt.xticks(np.arange(len(month_counts.index)))
plt.show()


# ### May Has the most Traffic With content activity 

# In[41]:


SB['Hour']=SB['datetime'].dt.hour


# ## Which Hour is Known to have more traffic?

# In[42]:


Hour_counts=SB['Hour'].value_counts()


# In[43]:


Hour_counts


# In[44]:


plt.bar(Hour_counts.index, Hour_counts.values , color='grey')

plt.xlabel('Hour')
plt.ylabel('Count')
plt.title('Count of each Hour')
plt.xticks(np.arange(len(Hour_counts.index)))

plt.show()


# ### Most users View Content at 6am

# ## Most Popular Type of content Among Users?

# In[45]:


Content_counts=SB['content_type'].value_counts()
Content_counts


# In[46]:


plt.bar(Content_counts.index, Content_counts.values , color='grey')

plt.xlabel('content_type')
plt.ylabel('Count')
plt.title('Count of each content_type')
plt.xticks(np.arange(len(Content_counts.index)))

plt.show()


# ### Photos Gain more traction on Social Buzz

# ## Content Sentiment in relation to content Type

# In[47]:


sentiment_counts = SB.groupby(['content_type', 'sentiment']).size().unstack(fill_value=0)
print(sentiment_counts)


# ### Photos Get the most views and  positive reactionsfollowed by videos then Gif's then Audios
# 

# In[48]:


SB


# In[106]:


SocialBuzzData=SB.to_csv("SocialBuzzData.csv")


# In[49]:


animal_reactions = SB[SB['category'] == 'Animals']
animal_reaction_counts = animal_reactions['reaction_type'].value_counts()
print(animal_reaction_counts)


# In[50]:


animal_reaction_counts.sum()


# In[105]:


plt.figure(figsize=(15, 6))
plt.plot(animal_reaction_counts.index, animal_reaction_counts.values, marker='o', linestyle='-')
plt.xlabel('reaction_type')
plt.ylabel('Count')
plt.title('Count of Reactions for Animal Category')



plt.show()


# ## Feature Engineering

# In[52]:


SB


# In[53]:


sentiment_mapping = {'Neutral': 0, 'Positive': 1, 'Negative': -1}
SB['sentiment_score'] = SB['sentiment'].map(sentiment_mapping)


# In[54]:


SB.tail()


# ## Sentiment Ratio

# In[55]:


SB['sentiment_score'].value_counts()


# In[56]:


import plotly.graph_objects as go

plot_data=[
    go.Pie(
        labels=("Negative", "Neutral","Positve"),
        values=SB['sentiment_score'].value_counts(),
        marker=dict(colors=["Red","Blue","Green"],
                      line=dict(color="white",
                                width=1.5)),
        
        rotation=90,
        hoverinfo= 'label+value+text',
        hole=.6)
]   

plot_layout = go.Layout(dict(title='Sentiment Ratio'))
fig = go.Figure(data=plot_data, layout=plot_layout)
fig.show()


# In[ ]:





# In[57]:


sentiment_mapping = {'Neutral': -1, 'Positive': 1, 'Negative': 0}
SB['sentiment_score'] = SB['sentiment'].map(sentiment_mapping)


# In[58]:


SB=SB.drop(SB[SB['sentiment_score'] == -1].index)                 


# In[59]:


import plotly.graph_objects as go

plot_data=[
    go.Pie(
        labels=("Negative","Positve"),
        values=SB['sentiment_score'].value_counts(),
        marker=dict(colors=["Red","Green"],
                      line=dict(color="white",
                                width=1.5)),
        
        rotation=90,
        hoverinfo= 'label+value+text',
        hole=.6)
]   


plot_layout = go.Layout(dict(title='Sentiment Ratio'))

fig = go.Figure(data=plot_data, layout=plot_layout)


fig.show()


# In[ ]:





# In[60]:


SB['content_type']=SB['content_type'].replace([True, False],[1,0])


# ## Further Data Cleaning

# In[61]:


SB.dropna()


# In[62]:


from sklearn.preprocessing import OneHotEncoder

ohe=OneHotEncoder()
ohe.fit_transform(SB[['content_type','category','reaction_type','score',]]).toarray()


# In[63]:


feature_arry=ohe.fit_transform(SB[['content_type','category','reaction_type']]).toarray()
ohe.categories_


# In[64]:


feature_df = pd.DataFrame(feature_arry, columns=ohe.get_feature_names_out(['content_type','category','reaction_type']))
feature_df


# In[65]:


SBM= pd.concat([SB, feature_df], axis=1)
SBM


# In[ ]:





# In[66]:


SBM=SBM.drop(columns=['content_type','category','reaction_type','sentiment','datetime'])
SBM


# ## Fixing Skewness In Data

# In[67]:


SBM.describe()


# In[68]:


parameters_with_std_gt_1 = SBM.describe().loc['std'] > 1


print("Parameters with standard deviation greater than 1:")
print(parameters_with_std_gt_1[parameters_with_std_gt_1].index.tolist())


# In[69]:


fig, axs = plt.subplots(nrows=2, figsize=(18, 20))

sns.distplot((SBM["Hour"].dropna()), ax=axs[0]) 
sns.distplot((SBM["month"].dropna()), ax=axs[1])
plt.show()


# In[ ]:





# In[70]:


SBM["Hour"] = np.log10(SBM["Hour"] + 1) 
SBM["month"] = np.log10(SBM["month"]+ 1)


# In[71]:


fig, axs = plt.subplots(nrows=2, figsize=(18, 20))

sns.distplot((SBM["Hour"].dropna()), ax=axs[0]) 
sns.distplot((SBM["month"].dropna()), ax=axs[1])
plt.show()


# In[72]:


##SB=SB.drop(columns=['content_id'])


# In[73]:


##SB.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[74]:


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

sns.regplot(y='month', x='sentiment_score', data=SBM,scatter_kws={'color': 'blue'}, line_kws={'color': 'red'}, ax=axs[0])
axs[0].set_title('Month vs Sentiment')


sns.regplot(y='Hour', x='sentiment_score', data=SBM,scatter_kws={'color': 'blue'}, line_kws={'color': 'red'}, ax=axs[1])
axs[1].set_title('Hour vs Sentiment')


# In[75]:


SBM.columns


# In[76]:


SBM=SBM.dropna()


# In[77]:


SBM


# ## Model Building

# In[ ]:





# In[78]:


from sklearn.model_selection import train_test_split
X =SBM.drop(columns= ['sentiment_score','content_id'], axis=1)
Y=SBM['sentiment_score']
X_train, X_tests, Y_train, Y_tests = train_test_split(X, Y, test_size=0.3, random_state=42)
print('Test set:', X_train.shape, Y_train.shape)
print('Test set:', X_tests.shape, Y_tests.shape)


# In[79]:


correlation = SBM.drop(columns= ['sentiment_score','content_id'], axis=1).corr()
plt.figure(figsize=(45, 45))
sns.heatmap(correlation, xticklabels=correlation.columns.values, yticklabels=correlation.columns.values, annot=True,
annot_kws={'size': 12})

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


# In[80]:


print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_tests.shape,  Y_tests.shape) 


# In[81]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.inspection import permutation_importance

from yellowbrick.classifier import ConfusionMatrix
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold


# In[82]:


def model_fit_predict(model, X, Y, X_predict):
    model.fit(X,Y)
    return model.predict(X_predict)
def acc_score(Y_true, Y_pred):
    return accuracy_score(Y_true, Y_pred)
def pre_score(Y_true,Y_pred):
    return precision_score(Y_true, Y_pred)
def f_score(Y_true, Y_pred):
    return f1_score(Y_true, Y_pred)


# In[83]:


model1 = RandomForestClassifier()

Y_pred_test = model_fit_predict(model1, X_train, Y_train, X_tests)


f1 = round(f1_score(Y_tests, Y_pred_test),2) 

acc = round(accuracy_score(Y_tests, Y_pred_test),2) 

pre = round(precision_score(Y_tests, Y_pred_test,) ,2) 

print(f"Accuracy, precision and f1-score for training data are {acc}, {pre} and {f1} respectively")


# In[84]:


from sklearn import metrics
predictions1 = model1.predict(X_tests)
tn, fp, fn, tp = metrics.confusion_matrix(Y_tests, predictions1).ravel()
Y_tests.value_counts()


print(f"True positives: {tp}") 
print(f"False positives: {fp}") 
print(f"True negatives: {tn}") 
print(f"False negatives: {fn}\n")

print(f"Accuracy: {metrics.accuracy_score(Y_tests, predictions1)}") 
print(f"Precision: {metrics.precision_score(Y_tests, predictions1)}") 
print(f"Recall: {metrics.recall_score(Y_tests, predictions1)}")


# In[85]:


from sklearn.metrics import confusion_matrix
from yellowbrick.classifier import ConfusionMatrix
cm1= ConfusionMatrix(model1, classes=[0,1])
cm1.fit(X_train, Y_train)

cm1.score(X_train, Y_train)


# In[86]:


feature_importances = pd.DataFrame({'features': X_train.columns,'importance': model1.feature_importances_}).sort_values(by='importance', ascending=True).reset_index()


# In[87]:


plt.figure(figsize=(15, 35))
plt.title('Feature Importances')
plt.barh(range(len(feature_importances)), feature_importances['importance'],color='b', align='center')
plt.yticks(range(len(feature_importances)), feature_importances['features']) 
plt.xlabel('Importance')
plt.show()


# In[88]:


def correlation (dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range (len (corr_matrix.columns)) :
        for j in range(i):
            if abs (corr_matrix.iloc [i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns [i] # getting the name of column
                col_corr.add(colname)
    return col_corr


# In[89]:


corr_features=correlation(X_train, 0.7)
len(set(corr_features))


# ### No features are highly correlated

# In[90]:


predictions = model1.predict(X_tests)
tn, fp, fn, tp = metrics.confusion_matrix(Y_tests, predictions).ravel()
Y_tests.value_counts()


# In[91]:


proba_predictions = model1.predict_proba(X_tests)
probabilities = proba_predictions[:,1]


# In[92]:


X_tests['sentiment_score'] = predictions.tolist()
X_tests['Sentiment_probability'] = probabilities.tolist()
X_tests.to_csv('sampled_data_with_predictions.csv')


# In[93]:


SD=pd.read_csv("sampled_data_with_predictions.csv")


# In[94]:


SD.info()


# In[95]:


import numpy as np
import matplotlib.pyplot as plt

# Count positive and negative sentiments for each column
audio_positives = SD[SD['content_type_Audio'] == 1]['sentiment_score'].count()
audio_negatives = SD[SD['content_type_Audio'] == 0]['sentiment_score'].count()

gif_positives = SD[SD['content_type_GIF'] == 1]['sentiment_score'].count()
gif_negatives = SD[SD['content_type_GIF'] == 0]['sentiment_score'].count()

photo_positives = SD[SD['content_type_Photo'] == 1]['sentiment_score'].count()
photo_negatives = SD[SD['content_type_Photo'] == 0]['sentiment_score'].count()

video_positives = SD[SD['content_type_Video'] == 1]['sentiment_score'].count()
video_negatives = SD[SD['content_type_Video'] == 0]['sentiment_score'].count()

# Define the columns and their corresponding positive and negative counts
columns = ['Audio', 'GIF', 'Photo', 'Video']
positives = [audio_positives, gif_positives, photo_positives, video_positives]
negatives = [audio_negatives, gif_negatives, photo_negatives, video_negatives]

# Define the width of the bars
bar_width = 0.35

# Set the positions for the bars
r1 = np.arange(len(columns))
r2 = [x + bar_width for x in r1]

# Create the grouped bar chart
plt.bar(r1, positives, color='b', width=bar_width, edgecolor='grey', label='Positive')
plt.bar(r2, negatives, color='r', width=bar_width, edgecolor='grey', label='Negative')

# Add counts to the bars
for i in range(len(r1)):
    plt.text(x=r1[i] - bar_width/2, y=positives[i] + 5, s=str(positives[i]), ha='center')
    plt.text(x=r2[i] - bar_width/2, y=negatives[i] + 5, s=str(negatives[i]), ha='center')

# Add labels and title
plt.xlabel('Content Type', fontweight='bold')
plt.ylabel('Sentiment Counts', fontweight='bold')
plt.xticks([r + bar_width/2 for r in range(len(columns))], columns)
plt.title('Positive and Negative Sentiment Counts for Content Types')
plt.legend()

# Show plot
plt.show()


# In[96]:


SB.count


# In[ ]:





# In[ ]:




