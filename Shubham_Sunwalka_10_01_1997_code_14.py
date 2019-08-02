#!/usr/bin/env python
# coding: utf-8

# # CHRISTIANO RONALDO

# # Loading The Dataset

# In[1]:


import pandas as pd 
import numpy as np                     # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt        # For plotting graphs 
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings                        # To ignore any warnings warnings.filterwarnings("ignore")


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# In[2]:


TrainDataPath = 'data.csv'
sample_submissionPath = 'sample_submission.csv'
# Loading the Training and Test Dataset
Train = pd.read_csv(TrainDataPath)
sample_submission = pd.read_csv(sample_submissionPath)


# In[3]:


Train_oroginal=Train.copy()


# In[4]:


print("Training Dataset Shape:")
print(Train.shape)
print("\n")
print("Training Dataset Columns/Features:")
print(Train.dtypes)
Train.head()


# In[5]:


print("Submission Dataset Shape:")
print(sample_submission.shape)
print("\n")
print("Submission Dataset Columns/Features:")
print(sample_submission.dtypes)
sample_submission.head()


# In[6]:


# checking missing data percentage in train data
total = Train.isnull().sum().sort_values(ascending = False)
percent = (Train.isnull().sum()/Train.isnull().count()*100).sort_values(ascending = False)
missing_TrainData  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_TrainData.head(30)


# # EDA(Exploratory Data Analysis)

# # Univariate Analysis

# In[7]:


plt.figure(figsize=(12,8))
sns.distplot(Train.match_event_id.values, bins=50, kde=False)
plt.xlabel('match_event_id', fontsize=12)
plt.show()


# In[8]:


plt.figure(figsize=(12,8))
sns.distplot(Train.location_x.values, bins=50, kde=False)
plt.xlabel('y value', fontsize=12)
plt.show()


# In[9]:


plt.figure(figsize=(12,8))
sns.distplot(Train.location_y.values, bins=50, kde=False)
plt.xlabel('y value', fontsize=12)
plt.show()


# In[10]:


plt.figure(figsize=(12,8))
sns.distplot(Train.distance_of_shot.values, bins=50, kde=False)
plt.xlabel('distance_of_shot', fontsize=12)
plt.show()


# In[11]:


plt.figure(figsize=(12,8))
sns.distplot(Train.remaining_sec.values, bins=50, kde=False)
plt.xlabel('remaining_sec', fontsize=12)
plt.show()


# In[12]:


plt.figure(figsize=(12, 6))
sns.countplot(Train["remaining_min"])
plt.title('remaining_min')
plt.show()


# In[13]:


plt.figure(figsize=(12, 6))
sns.countplot(Train["power_of_shot"])
plt.title('power_of_shot')
plt.show()


# In[14]:


plt.figure(figsize=(12, 6))
sns.countplot(Train["knockout_match"])
plt.title('knockout_match')
plt.show()


# In[15]:


plt.figure(figsize=(25, 6))
sns.countplot(Train["game_season"])
plt.title('game_season')
plt.show()


# In[16]:


plt.figure(figsize=(25, 6))
sns.countplot(Train["remaining_sec"])
plt.title('remaining_sec')
plt.show()


# In[17]:


plt.figure(figsize=(30, 6))
sns.countplot(Train["distance_of_shot"])
plt.title('distance_of_shot')
plt.show()


# In[18]:


plt.figure(figsize=(30, 6))
sns.countplot(Train["is_goal"])
plt.title('is_goal')
plt.show()


# In[19]:


plt.figure(figsize=(30, 6))
sns.countplot(Train["area_of_shot"])
plt.title('area_of_shot')
plt.show()


# In[20]:


plt.figure(figsize=(30, 6))
sns.countplot(Train["shot_basics"])
plt.title('shot_basics')
plt.show()


# In[21]:


plt.figure(figsize=(30, 6))
sns.countplot(Train["range_of_shot"])
plt.title('range_of_shot')
plt.show()


# In[22]:


plt.figure(figsize=(30, 6))
sns.countplot(Train["team_name"])
plt.title('team_name')
plt.show()


# In[23]:


plt.figure(figsize=(30, 6))
sns.countplot(Train["date_of_game"])
plt.title('date_of_game')
plt.show()


# In[24]:


plt.figure(figsize=(30, 6))
sns.countplot(Train["home/away"])
plt.title('home/away')
plt.show()


# In[25]:


plt.figure(figsize=(30, 6))
sns.countplot(Train["shot_id_number"])
plt.title('shot_id_number')
plt.show()


# In[26]:


plt.figure(figsize=(30, 6))
sns.countplot(Train["lat/lng"])
plt.title('lat/lng')
plt.show()


# In[27]:


plt.figure(figsize=(30, 6))
sns.countplot(Train["type_of_shot"])
plt.title('type_of_shot')
plt.show()


# In[28]:


plt.figure(figsize=(30, 6))
sns.countplot(Train["type_of_combined_shot"])
plt.title('type_of_combined_shot')
plt.show()


# In[29]:


plt.figure(figsize=(30, 6))
sns.countplot(Train["match_id"])
plt.title('match_id')
plt.show()


# In[30]:


plt.figure(figsize=(30, 6))
sns.countplot(Train["team_id"])
plt.title('team_id')
plt.show()


# # Bivariate Analysis

# In[31]:


plt.figure(figsize=(30,30))

sns.pairplot(Train, diag_kind='kde');


# In[32]:


#correlation matrix
plt.figure(figsize=(20, 10))

vg_corr = Train.corr()
sns.heatmap(vg_corr, 
            xticklabels = vg_corr.columns.values,
            yticklabels = vg_corr.columns.values,
            annot = True);


# In[33]:


# categorical vs Categorical
#Train.plot.bar(stacked=True)
Married=pd.crosstab(Train['power_of_shot'],Train['is_goal']) 
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(20,4))
plt.show() 


# In[34]:


# categorical vs Categorical
#Train.plot.bar(stacked=True)
Married=pd.crosstab(Train['knockout_match'],Train['is_goal']) 
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(20,4))
plt.show() 


# In[35]:


# categorical vs Categorical
#Train.plot.bar(stacked=True)
Married=pd.crosstab(Train['game_season'],Train['is_goal']) 
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(20,4))
plt.show() 


# In[36]:


# categorical vs Categorical
#Train.plot.bar(stacked=True)
Married=pd.crosstab(Train['area_of_shot'],Train['is_goal']) 
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(20,4))
plt.show() 


# In[37]:


# categorical vs Categorical
#Train.plot.bar(stacked=True)
Married=pd.crosstab(Train['shot_basics'],Train['is_goal']) 
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(20,4))
plt.show() 


# In[38]:


# categorical vs Categorical
#Train.plot.bar(stacked=True)
Married=pd.crosstab(Train['range_of_shot'],Train['is_goal']) 
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(20,4))
plt.show() 


# In[39]:


# categorical vs Categorical
#Train.plot.bar(stacked=True)
Married=pd.crosstab(Train['type_of_shot'],Train['is_goal']) 
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(20,4))
plt.show() 


# In[40]:


# categorical vs Categorical
#Train.plot.bar(stacked=True)
Married=pd.crosstab(Train['type_of_combined_shot'],Train['is_goal']) 
Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(20,4))
plt.show() 


# In[41]:


# Renaming Columns which are repeating with Different Names for analyzation
Train.rename(columns={'remaining_min.1': 'remaining_minONE', 'power_of_shot.1': 'power_of_shotONE' ,'knockout_match.1': 'knockout_matchONE','remaining_sec.1': 'remaining_secONE', 'distance_of_shot.1':'distance_of_shotONE'}, inplace=True)


# In[42]:


Train.describe()


# In[43]:


# Function to preprocess home/away data
def preprocess_home_away(x):
    try:
        t = x.split(' ')
    except:
        return None
    return t[0]+' vs. '+t[2]


# In[44]:


Train['home/away'] = Train['home/away'].apply(lambda x: preprocess_home_away(x))


# # Missing values Handling

# In[45]:


Train.shape


# In[46]:


for i in range(len(Train)):
    if(type(Train.loc[i ,'type_of_combined_shot']) == float):
        Train.loc[i ,'type_of_combined_shot'] = 'shot - ' + str(int(int(Train.loc[i ,'type_of_shot'].split('-')[1])/10))


# In[47]:


for i in range(len(Train)):
    if(type(Train.loc[i ,'type_of_shot']) == float):
        Train.loc[i ,'type_of_shot'] = 'shot - ' + str(int(Train.loc[i ,'type_of_combined_shot'].split('-')[1])*10+5)


# In[48]:


# splitting the numerical feature i.e. 45 from shot-45
Train['type_of_shot'] = Train['type_of_shot'].apply(lambda x : int(x.split('-')[1]))


# In[49]:


for i in range(len(Train)):
    if(np.isnan(Train.loc[5 ,'power_of_shot'])):
        Train.loc[i ,'power_of_shot'] = int(Train.loc[i ,'power_of_shotONE']/10)


# In[50]:


Train['power_of_shot'].fillna(Train['power_of_shot'].mode()[0], inplace=True)


# In[51]:


for i in range(len(Train)):
    if(np.isnan(Train.loc[5 ,'remaining_min'])):
        Train.loc[i ,'remaining_min'] = int(Train.loc[i ,'remaining_minONE']/10)


# In[52]:


Train['remaining_min'].fillna(Train['remaining_min'].mode()[0], inplace=True)


# In[53]:


# filling knockout_match


# In[54]:


lis_knockout = Train.index.tolist()# put index in a list
for i in range(len(lis_knockout)):
    if(np.isnan(Train.loc[lis_knockout[i],'knockout_match'])):
        Train.loc[lis_knockout[i],'knockout_match']= Train.loc[lis_knockout[i], 'knockout_matchONE']


# In[55]:


# now filling remaining missing values with median/mode
Train['knockout_match'].fillna(Train['knockout_match'].mode()[0], inplace=True)


# In[56]:


# distance_of_shot
lis_knockout = Train.index.tolist()# put index in a list
for i in range(len(lis_knockout)):
    if(np.isnan(Train.loc[lis_knockout[i],'distance_of_shot'])):
        Train.loc[lis_knockout[i],'distance_of_shot']= round(Train.loc[lis_knockout[i], 'distance_of_shotONE'])


# In[57]:


Train['distance_of_shot'].fillna(Train['distance_of_shot'].median(), inplace=True)


# In[58]:


# filling  remaining_sec


# In[59]:


lis_knockout = Train.index.tolist()# put index in a list
for i in range(len(lis_knockout)):
    if(np.isnan(Train.loc[lis_knockout[i],'remaining_sec'])):
        Train.loc[lis_knockout[i],'remaining_sec']= Train.loc[lis_knockout[i], 'remaining_secONE']


# In[60]:


Train['remaining_sec'].fillna(Train['remaining_sec'].median(), inplace=True)


# In[61]:


Train['location_x'].fillna(Train['location_x'].mean(), inplace=True)


# In[62]:


Train['home/away'].fillna(Train['home/away'].mode()[0], inplace=True)


# In[63]:


Train['area_of_shot'].fillna(Train['area_of_shot'].mode()[0], inplace=True)


# In[64]:


Train['team_name'].fillna(Train['team_name'].mode()[0], inplace=True)


# In[65]:


Train['location_y'].fillna(Train['location_y'].median(), inplace=True)


# In[66]:


Train[['year','month','date']] = Train.date_of_game.str.split('-', expand=True)

Train['year'].fillna(Train['year'].mode()[0], inplace=True)
Train['month'].fillna(Train['month'].mode()[0], inplace=True)
Train['date'].fillna(Train['date'].mode()[0], inplace=True)


# In[67]:


Train['match_event_id'].fillna(Train['match_event_id'].median(), inplace=True)


# In[68]:


Train['range_of_shot'].fillna(Train['range_of_shot'].mode()[0], inplace=True)


# In[69]:


Train['lat/lng'].fillna(Train['lat/lng'].mode()[0], inplace=True)


# In[70]:


Train['shot_basics'].fillna(Train['shot_basics'].mode()[0], inplace=True)


# In[71]:


Train['game_season'].fillna(Train['game_season'].mode()[0], inplace=True)


# In[72]:


Train=Train.drop(['remaining_minONE','power_of_shotONE','knockout_matchONE','remaining_secONE','distance_of_shotONE'], axis=1)


# In[73]:


Train=Train.drop(['date_of_game'], axis=1)


# In[74]:


# checking missing data percentage in train data
total = Train.isnull().sum().sort_values(ascending = False)
percent = (Train.isnull().sum()/Train.isnull().count()*100).sort_values(ascending = False)
missing_TrainData  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_TrainData.head(30)


# In[75]:


Train.head()


# In[76]:


Train.columns


# In[77]:


Train.shape


# In[78]:


# Delete columns at index 1 & 2
Train = Train.drop([Train.columns[0]] ,  axis='columns')


# In[79]:


Train.head()


# In[80]:


# drop team_name and team_id as they are same throughout individually.
Train = Train.drop(['team_name','team_id'] ,  axis=1)


# In[81]:


Train.columns


# In[82]:


# Label Encoding game_season, area_of_shot, shot_basics, range_of_shot, home/away, lat/lng, type_of_combined_shot
# respectively
labelencoder = LabelEncoder()

labelencoder.fit(Train.iloc[:,6].values)
Train.iloc[:,6]=labelencoder.transform(Train.iloc[:,6])


labelencoder.fit(Train.iloc[:,10].values)
Train.iloc[:,10]=labelencoder.transform(Train.iloc[:,10])

labelencoder.fit(Train.iloc[:,11].values)
Train.iloc[:,11]=labelencoder.transform(Train.iloc[:,11])

labelencoder.fit(Train.iloc[:,12].values)
Train.iloc[:,12]=labelencoder.transform(Train.iloc[:,12])

labelencoder.fit(Train.iloc[:,13].values)
Train.iloc[:,13]=labelencoder.transform(Train.iloc[:,13])

labelencoder.fit(Train.iloc[:,15].values)
Train.iloc[:,15]=labelencoder.transform(Train.iloc[:,15])

labelencoder.fit(Train.iloc[:,17].values)
Train.iloc[:,17]=labelencoder.transform(Train.iloc[:,17])


# In[83]:


Train.head()


# In[84]:


Train.shape


# In[85]:


# Filling shot_id_number with previous value + 1 (Series of Natural Numbers)
lis = Train[Train['shot_id_number'].isnull()].index.tolist()
for i in range(len(lis)):
    Train.loc[lis[i],'shot_id_number']= Train.loc[lis[i]-1, 'shot_id_number']+1


# In[86]:


# checking missing data percentage in train data
total = Train.isnull().sum().sort_values(ascending = False)
percent = (Train.isnull().sum()/Train.isnull().count()*100).sort_values(ascending = False)
missing_TrainData  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_TrainData.head(30)


# In[87]:


lis2 = Train[Train['is_goal'].isnull()].index.tolist()
#for i in range(len(lis2)):
#    Test1=Train.loc[lis2[i],:] 


# In[88]:


#Test1.count()


# 'match_event_id','location_x','location_y','remaining_min','power_of_shot','knockout_match',
#               'game_season','remaining_sec','distance_of_shot','area_of_shot','shot_basics','range_of_shot','team_name','date_of_game',
#               'home/away','shot_id_number','lat/lng','type_of_shot','type_of_combined_shot','match_id',
#               'team_id'

# In[89]:


df3=Train


# In[90]:


df4=df3.iloc[lis2,:]


# In[91]:


df4.head()# contains rowws where is_goal is missing


# In[92]:


df4.to_csv('Test.csv')


# In[93]:


TestDataPath = 'Test.csv'
# Loading the Training and Test Dataset
Test = pd.read_csv(TestDataPath)
Test_original=Test.copy()


# In[94]:


Test.head()


# In[95]:


# Delete columns at index 1 & 2
Test = Test.drop([Test.columns[0]] ,  axis='columns')


# In[96]:


# checking missing data percentage in train data
total = Test.isnull().sum().sort_values(ascending = False)
percent = (Test.isnull().sum()/Test.isnull().count()*100).sort_values(ascending = False)
missing_TrainData  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_TrainData.head(30)


# In[97]:


X=Train
#Y=Train


# In[98]:


X.head()


# In[99]:


X.columns


# In[100]:


#X['date_of_game'].dt.strftime("%Y%m%d").astype(int)
#X['date_of_game']
#X['date_of_game'].str.replace("-","")
#X['date_of_game'] = pd.to_numeric(X.date_of_game.str.replace('-',''))
#Test['date_of_game'] = pd.to_numeric(Test.date_of_game.str.replace('-',''))


# In[101]:


X.head()


# In[102]:


#X=pd.DataFrame(X)
#X=X.drop(['game_season'], axis=1)
#Test=Test.drop(['game_season'], axis=1)


# In[103]:


X.head()


# In[104]:


Train.head()


# In[105]:


# checking missing data percentage in train data
total = X.isnull().sum().sort_values(ascending = False)
percent = (X.isnull().sum()/X.isnull().count()*100).sort_values(ascending = False)
missing_TrainData  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_TrainData.head(30)


# In[106]:


# dropping is_goal in X i.e Train dataset wherever it is null
X_demo=X


# Dropping the rows where there is not NULL Value or missing value 

# In[107]:


import numpy as np

X_demo = X_demo[np.isfinite(X_demo['is_goal'])]
#Test = Test[np.isfinite(Test['is_goal'])]


# In[108]:


Test.head()


# In[109]:


X_demo.head()


# In[110]:


# checking missing data percentage in train data
total = X_demo.isnull().sum().sort_values(ascending = False)
percent = (X_demo.isnull().sum()/X_demo.isnull().count()*100).sort_values(ascending = False)
missing_TrainData  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_TrainData.head(30)


# In[111]:


X=X_demo


# In[112]:


X.to_csv('v1.csv')


# In[113]:


X.head()


# Making y DataFrame for Model Feeding

# In[114]:


y=pd.DataFrame(X['is_goal'])


# In[115]:


y.head()


# In[116]:


y.shape


# In[117]:


X.head()


# In[118]:


X['is_goal'].head()


# In[119]:


# Drop is_goal from both datasets

X=X.drop(['is_goal'],axis=1)
Test=Test.drop(['is_goal'],axis=1)


# In[120]:


X.head()


# In[121]:


# Now i have X and y values for train teest split


# In[122]:


X.columns


# In[123]:


Test.columns


# In[124]:


X.shape


# In[125]:


Test.shape


# In[126]:


X1=X
Test1=Test


# In[127]:


X1.shape


# # Prediction

# In[128]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X1,y,test_size=0.2,random_state=0)


# In[129]:


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train_ = sc_X.fit_transform(X_train)
X_test_ = sc_X.transform(X_test)
Test_ = sc_X.transform(Test)


# In[130]:


Test.shape


# In[131]:


X_train['year'] = X_train['year'].astype(int)
X_train['month'] = X_train['month'].astype(int)
X_train['date'] = X_train['date'].astype(int)


# In[132]:


Test['year'] = Test['year'].astype(int)
Test['month'] = Test['month'].astype(int)
Test['date'] = Test['date'].astype(int)


# In[133]:


X_test['year'] = X_test['year'].astype(int)
X_test['month'] = X_test['month'].astype(int)
X_test['date'] = X_test['date'].astype(int)


# In[134]:


# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(solver='lbfgs',random_state=0,C=10)
classifier.fit(X, y)


# #  Fitting Logistic Regression to the Training set

# In[135]:


y_pred=classifier.predict_proba(X_test)


# In[136]:


y_pred1=classifier.predict_proba(Test)


# In[137]:


preds=y_pred1[:,1]


# In[138]:


preds = np.round(preds,2)


# # Extracting corresponding shot_id_number from sample submission file as dataset size is different for sample_submission.csv and Test.csv

# In[139]:


ans=pd.DataFrame(y_pred1)


# In[140]:


ans.head()


# In[141]:


# Delete columns at index 1 & 2
ans = ans.drop([ans.columns[0]] ,  axis='columns')


# In[142]:


ans.head()


# In[143]:


ans.columns = ['is_goal']


# In[144]:


ans.head()


# In[145]:


# map with shot_id_number


# In[146]:


G=Test_original


# In[147]:


G.head()


# In[148]:


shot_id_list=G['shot_id_number'].to_list()


# In[149]:


shot_id_list


# In[150]:


shot_id_list=pd.DataFrame(shot_id_list)


# In[151]:


shot_id_list.head()


# In[152]:


shot_id_list.columns=['shot_id_number']


# In[153]:


shot_id_list.head()


# In[154]:


shot_id_list.set_index('shot_id_number',inplace=True)


# In[155]:


shot_id_list.head()


# In[156]:


#generating submission csv
submission = pd.DataFrame({'shot_id_number':shot_id_list.index,'is_goal':preds})
#save the file to your directory
submission.to_csv('submission_prob.csv',index=False)


# In[157]:


submission.head()


# In[158]:


submission.set_index('shot_id_number',inplace=True)


# In[159]:


submission.head()


# In[160]:


sample_submission.head()


# In[161]:


sample_submission=pd.read_csv('sample_submission.csv')
sample_submission.set_index('shot_id_number',inplace=True)


# In[162]:


sample_submission.head()


# In[163]:


a = sample_submission.index.values.tolist()


# In[164]:


final_sub_df = submission.loc[a,:]


# In[167]:


final_sub_df.to_csv('Shubham_Sunwalka_10_01_1997_prediction_14.csv',index = True)


# In[168]:


len(final_sub_df)


# # The End
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




