#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
import string


# In[2]:


import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB

from gensim.models import Word2Vec
from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')


# In[3]:


import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


# In[36]:


st.set_page_config(page_title ='Indeed Data Science Job Postings')
st.header('Data Science Job Posts')
st.subheader('Team 67')


# In[4]:


us_state_abbrev_reverse = {
    'AL': 'Alabama',
    'AK': 'Alaska',
    'AZ': 'Arizona',
    'AR': 'Arkansas',
    'CA': 'California',
    'CO': 'Colorado',
    'CT': 'Connecticut',
    'DC': 'District of Columbia',
    'DE': 'Delaware',
    'FL': 'Florida',
    'GA': 'Georgia',
    'HI': 'Hawaii',
    'ID': 'Idaho',
    'IL': 'Illinois',
    'IN': 'Indiana',
    'IA': 'Iowa',
    'KS': 'Kansas',
    'KY': 'Kentucky',
    'LA': 'Louisiana',
    'ME': 'Maine',
    'MD': 'Maryland',
    'MA': 'Massachusetts',
    'MI': 'Michigan',
    'MN': 'Minnesota',
    'MS': 'Mississippi',
    'MO': 'Missouri',
    'MT': 'Montana',
    'NE': 'Nebraska',
    'NV': 'Nevada',
    'NH': 'New Hampshire',
    'NJ': 'New Jersey',
    'NM': 'New Mexico',
    'NY': 'New York',
    'NC': 'North Carolina',
    'ND': 'North Dakota',
    'OH': 'Ohio',
    'OK': 'Oklahoma',
    'OR': 'Oregon',
    'PA': 'Pennsylvania',
    'PR': 'Puerto Rico',
    'RI': 'Rhode Island',
    'SC': 'South Carolina',
    'SD': 'South Dakota',
    'TN': 'Tennessee',
    'TX': 'Texas',
    'UT': 'Utah',
    'VT': 'Vermont',
    'VA': 'Virginia',
    'WA': 'Washington',
    'WV': 'West Virginia',
    'WI': 'Wisconsin',
    'WY': 'Wyoming',
}

#Northern Mariana Islands':'MP', 'Palau': 'PW', 'Puerto Rico': 'PR', 'Virgin Islands': 'VI', 'District of Columbia': 'DC'


# In[5]:


us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

#'Northern Mariana Islands':'MP', 'Palau': 'PW', 'Puerto Rico': 'PR', 'Virgin Islands': 'VI', 'District of Columbia': 'DC'
#abbrev_us_state = dict(map(reversed, us_state_abbrev.items()))


# In[6]:


# Load CSV file into a DataFrame
#ds_jobs_df = pd.read_csv('job_postings_June_2021.csv')
ds_jobs_df = pd.read_csv('Indeed_Data_Science_Job_Postings_June_2021_DS4A_Project.csv')


# In[7]:


#drop duplicates
ds_jobs = ds_jobs_df.copy()
ds_jobs.drop_duplicates(inplace=True)


# In[8]:


ds_jobs.dropna(inplace = True)


# #### Data Cleaning

# In[9]:


def clean_city(df):
    new = df['location'].str.split(",", expand = True)
    n1 = new[0].str.split("+", expand = True)
    new['city'] = n1[0]
    n4 = new['city'].str.split("•", expand = True)
    job_data['city'] = n4[0]
    #return job_data['city']


# In[10]:


def clean_location(df):
    new = df['location'].str.split(",", expand = True)
    
    #clean_city(df)
    #Clean City
    n1 = new[0].str.split("+", expand = True)
    new['city'] = n1[0]
    n4 = new['city'].str.split("•", expand = True)
    new['city'] = n4[0]
    df['city'] = n4[0]
    
    #Clean State
    n2 = new[1].str.split(" ", expand = True)
    n2 = n2[1].str.replace('•', "+")
    n2 = n2.str.replace('+', " ")
    n2 = n2.str.split(" ", expand = True)
    new['State'] = n2[0]
    new['State'] = new['State'].replace(us_state_abbrev_reverse)
    new['State'].fillna(value='empty', inplace=True)
    new['State'].replace('empty', 'Same_with_City', inplace=True)
    
    new['State'] = np.where(new['State'] == 'Same_with_City', new['city'], new['State'])
    df['State'] = new['State']
    
    df['location'] = df[['city', 'State']].apply(lambda x: ', '.join(x), axis=1)
    
    return df


# In[11]:


jobs = clean_location(ds_jobs)


# In[12]:


def clean_text(text):
    text = str(text)
    text = text.replace("\n"," ")
    text = text.lower()
    #text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    return text.lower()


# In[13]:


jobs["clean_description"] = jobs["description"].apply(clean_text)


# #### Analysis of Job Titles

# In[14]:


Total_num_job_titles = jobs['title'].nunique()
print('Total number of titles related to job postings: ', Total_num_job_titles)


# In[15]:


#Most common roles across companies
Most_common_job_title = jobs.groupby(['title'])['company'].count()
Most_common_job_title = Most_common_job_title.reset_index()
Most_common_job_title = Most_common_job_title.sort_values(['company'], ascending=False)
Most_common_job_title = Most_common_job_title.head(25)
Most_common_job_title


# In[16]:


# Plot graph for top most offered roles
fig,ax=plt.subplots(figsize=(15,6))
ax=sns.barplot(x="title", y="company", data = Most_common_job_title)    
ax.set_xticklabels(Most_common_job_title['title'], rotation=90)
ax.set_xlabel('MOST WANTED JOB ROLES', fontsize=20, color='blue')
ax.set_ylabel('NO OF ROLES ACROSS INDUSTRY', fontsize=12,color='blue')


# #### Categorizing Job Titles

# In[17]:


#There are so many job profiles in the given dataset so lets Categories them into 5; Data Scientist, Machine Learning Engineer, Data Analyst, Data Science Manager and Others

# Creating only 5 datascience roles among all
data = jobs.copy()

#data.dropna(subset=['title'], how='all', inplace = True)
data['position']= [x.upper() for x in data['title']]
data['description'] = [x.upper() for x in data['description']]

data.loc[data.position.str.contains("SCIENTIST"), 'position'] = 'Data Scientist'

data.loc[data.position.str.contains('ENGINEER'),'position']='Machine Learning Engineer'
data.loc[data.position.str.contains('PRINCIPAL STATISTICAL PROGRAMMER'),'position']='Machine Learning Engineer'

data.loc[data.position.str.contains('PROGRAMMER'),'position']='Machine Learning Engineer'
data.loc[data.position.str.contains('DEVELOPER'),'position']='Machine Learning Engineer'

data.loc[data.position.str.contains('ANALYST'), 'position'] = 'Data Analyst'
data.loc[data.position.str.contains('STATISTICIAN'), 'position'] = 'Data Analyst'

data.loc[data.position.str.contains('MANAGER'),'position']='Data Science Manager'

data.loc[data.position.str.contains('CONSULTANT'),'position']='Data Science Manager'
data.loc[data.position.str.contains('DATA SCIENCE'),'position']='Data Science Manager'
data.loc[data.position.str.contains('DIRECTOR'),'position']='Data Science Manager'

#data.position=data[(data.position == 'Data Scientist') | (data.position == 'Data Analyst') | (data.position == 'Machine Learning Engineer') | (data.position == 'Data Science Manager')]
#data.position=['Others' if x is np.nan else x for x in data.position]

position=data.groupby(['position'])['company'].count()   
position=position.reset_index(name='company')
position=position.sort_values(['company'],ascending=False)

#print('Here is  the count of each new roles we created :', '\n\n', position)


# In[18]:


t_roles = position.head(5)
t_roles.plot.bar(x='position', y='company')


# #### Analyzing Job Descriptions

# In[39]:


data["clean_description"] = data["description"].apply(clean_text)


# In[40]:


sentences = []
sent_word_sets = []
for row in data.iterrows():
    desc = row[1].clean_description
    word_tokens = nltk.word_tokenize(desc)
    sentences.append(word_tokens)
    sent_word_sets.append(set(word_tokens))


# In[41]:


model = Word2Vec(sentences=sentences, window=5, min_count=10, workers=4)#,size=100


# In[21]:


possible_words = set()
similar_words = model.wv.most_similar('bachelor', topn=30)
for tup in similar_words:
    possible_words.add(tup[0])
#similar_words


# In[22]:


similar_words = model.wv.most_similar('masters', topn=30)
for tup in similar_words:
    possible_words.add(tup[0])
#similar_words


# In[23]:


similar_words = model.wv.most_similar('phd', topn=30)
for tup in similar_words:
    possible_words.add(tup[0])
#similar_words


# In[24]:


bachelor_list = ['bs','b.s','bsc','bs/ms','bachelor','ba/bs','b.s.','bs/ms/phd','bachelors','ba','bs/ba','undergraduate']
master_list = ['masters','master','bs/ms','m.s.','m.s','msc','bs/ms/phd','ms','md/phd','ms/phd','postgraduate']
phd_list = ['phd','ph.d.','ph.d','bs/ms/phd','md/phd','ms/phd','doctoral','postgraduate','doctorate']


# In[43]:


jobs["sent_word_sets"] = sent_word_sets
data["sent_word_sets"] = sent_word_sets


# In[44]:


def has_qual(word_set,qual_list):
    for word in qual_list:
        if word in word_set: #we want this part to be o(1) since qual_list is much shorter than word_set
            return True
    return False


# In[45]:


jobs["bachelors"] = jobs["sent_word_sets"].apply(lambda x: has_qual(x,bachelor_list))
jobs["masters"] = jobs["sent_word_sets"].apply(lambda x: has_qual(x,master_list))
jobs["phd"] = jobs["sent_word_sets"].apply(lambda x: has_qual(x,phd_list))


# In[46]:


print("Number of jobs with descriptions have bachelor:", jobs["bachelors"].sum())
print("Number of jobs with descriptions have masters:", jobs["masters"].sum())
print("Number of jobs with descriptions have phd:", jobs["phd"].sum())


# In[47]:


def get_minimum(hasBsc,hasMsc,hasPhd):
    """
    returns minimum qualification if any
    """
    if hasBsc:
        return "Bachelors"
    
    elif hasMsc:
        return "Masters"
    
    elif hasPhd:
        return "Phd"
    
    else:
        return "No qualifications stated"


# In[48]:


jobs["min_qualification"] = jobs.apply(lambda x: get_minimum(x.bachelors,x.masters,x.phd),axis=1)


# In[49]:


value_counts = jobs["min_qualification"].value_counts()


# In[50]:


print("The number jobs that require a minimum of Bachelors are",value_counts["Bachelors"])
print("The number jobs that require a minimum of Masters are",value_counts["Masters"])
print("The number jobs that require a minimum of Phd are",value_counts["Phd"])
print("The number jobs that does not state education require",value_counts["No qualifications stated"])
print("The total number of jobs are", jobs.shape[0])


# #### Word Cloud of Job Descriptions

# In[51]:


lemmatizer = WordNetLemmatizer()
def clean_position(text):    
    text = re.sub(r"[^A-Za-z0-9]", " ", str(text)).lower()
    text_tokens = nltk.word_tokenize(text)
    text_lemmatized = [lemmatizer.lemmatize(word) for word in text_tokens]
    return " ".join(text_lemmatized)

data["clean_position"] = data.title.apply(clean_position)


# In[52]:


lemmatizer = WordNetLemmatizer()
def clean_position(text):    
    text = re.sub(r"[^A-Za-z0-9]", " ", str(text)).lower()
    text_tokens = nltk.word_tokenize(text)
    text_lemmatized = [lemmatizer.lemmatize(word) for word in text_tokens]
    return " ".join(text_lemmatized)

data["clean_position"] = data.title.apply(clean_position)


# In[53]:


def has_qual(word_set,qual_list):
    for word in qual_list:
        if word in word_set: #we want this part to be o(1) since qual_list is much shorter than word_set
            return True
    return False


# In[54]:


data["bachelors"] = data["sent_word_sets"].apply(lambda x: has_qual(x,bachelor_list))
data["masters"] = data["sent_word_sets"].apply(lambda x: has_qual(x,master_list))
data["phd"] = data["sent_word_sets"].apply(lambda x: has_qual(x,phd_list))


# In[56]:


data["min_qualification"] = data.apply(lambda x: get_minimum(x.bachelors,x.masters,x.phd),axis=1)


# In[57]:


bsc_string = " ".join(data["clean_position"][data["min_qualification"]=="Bachelors"])
msc_string = " ".join(data["clean_position"][data["min_qualification"]=="Masters"])
phd_string = " ".join(data["clean_position"][data["min_qualification"]=="Phd"])


# In[58]:


wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', max_words=100).generate(bsc_string)
plt.clf()
plt.title("Bachelor word cloud")
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[59]:


wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', max_words=100).generate(msc_string)
plt.clf()
plt.title("Masters word cloud")
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[60]:


wordcloud = WordCloud(stopwords=STOPWORDS, background_color='black', max_words=100).generate(phd_string)
plt.clf()
plt.title("Phd word cloud")
plt.imshow(wordcloud)
plt.axis('off')
plt.show()


# In[61]:


data_qual = data.groupby('min_qualification').size()


# In[62]:


data_qual.plot.bar()


# #### Understanding Degree Requirements - Most sought Degree Programs

# In[67]:


degree = data[data['clean_description'].str.contains('degree')]


# In[80]:


def degree_extraction(row):
    d = ','.join([a for a in row.split('\n') if 'DEGREE' in a][0].split('DEGREE')[1:])
    return d


# In[71]:


degree['description_majors'] = degree['description'].apply(degree_extraction)


# In[69]:


#degree['description'].apply(degree_extraction)


# In[72]:


degree_df = degree[degree['description_majors'].notnull()]


# In[73]:


relevant_degrees = ['computer science', 'Mathematics','statistics', 'analytics', 'data analytics', 'supply_chain', 
                    'finance', 'economics', 'business', 'accounting', 'logistics', 'data_science',
                    'engineering', 'physics', 'operations_research']


# In[74]:


degree_df['description_majors']
#d = degree_df[degree_df['description_majors'].str.contains('COMPUTER SCIENCE')]
comp_sci = degree_df[degree_df['description_majors'].str.contains('COMPUTER SCIENCE')]['description_majors'].count()
Mathematics = degree_df[degree_df['description_majors'].str.contains('MATHEMATICS')]['description_majors'].count()
statistics = degree_df[degree_df['description_majors'].str.contains('STATISTICS')]['description_majors'].count()
analytics = degree_df[degree_df['description_majors'].str.contains('ANALYTICS')]['description_majors'].count()
data_analytics = degree_df[degree_df['description_majors'].str.contains('DATA ANALYTICS')]['description_majors'].count()
supply_chain = degree_df[degree_df['description_majors'].str.contains('SUPPLY CHAIN')]['description_majors'].count()
finance = degree_df[degree_df['description_majors'].str.contains('FINANCE')]['description_majors'].count()
economics = degree_df[degree_df['description_majors'].str.contains('ECONOMICS')]['description_majors'].count()
business = degree_df[degree_df['description_majors'].str.contains('BUSINESS')]['description_majors'].count()
accounting = degree_df[degree_df['description_majors'].str.contains('ACCOUNTING')]['description_majors'].count()
logistics = degree_df[degree_df['description_majors'].str.contains('LOGISTICS')]['description_majors'].count()
data_science = degree_df[degree_df['description_majors'].str.contains('DATA SCIENCE')]['description_majors'].count()
physics = degree_df[degree_df['description_majors'].str.contains('PHYSICS')]['description_majors'].count()
operations_research = degree_df[degree_df['description_majors'].str.contains('OPERATIONS RESEARCH')]['description_majors'].count()
engineering = degree_df[degree_df['description_majors'].str.contains('ENGINEERING')]['description_majors'].count()                                                                                     


# In[75]:


jobs_covered = set()
for degree in relevant_degrees:
    jobs_covered = jobs_covered | set(degree_df[degree_df['description_majors'].str.contains(degree.upper())].index.to_list())


# In[76]:


majors_df = pd.DataFrame({'majors': [comp_sci, Mathematics, statistics, analytics, data_analytics, supply_chain, finance, economics, business, accounting, logistics, data_science, engineering, physics, operations_research]})
#columns=['comp_sci', 'Mathematics','statistics', 'supply_chain', 'finance', 'logistics', 'data_science','physics', 'operations_research']
major = majors_df.transpose()
major.columns = relevant_degrees 
major.transpose()


# In[77]:


major.transpose().plot.bar()


# In[82]:

m = major.transpose()
m.reset_index()


# In[85]:


mains = m.reset_index()

mains = mains.rename(columns={'index':'programs'})


# In[78]:


m = major.transpose()
m.plot.pie(subplots=True,y='majors', figsize=(15,8))
plt.legend(bbox_to_anchor=(1, 1), loc='best', borderaxespad=0.5)


pie_chart = px.pie(mains, title = 'Programs Required',
                values = 'majors', names ='programs')
                
st.plotly_chart(pie_chart)


# #### Extras

# In[ ]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

#Define the word cloud function with a max of 200 words
def plot_wordcloud(text, mask=None, max_words=200, max_font_size=120, figure_size=(15,15), 
                   title = None, title_size=20, image_color=False):
    stopwords = set(STOPWORDS)
    #define additional stop words that are not contained in the dictionary
    #more_stopwords = {'one', 'object', 'nTHE', 'Name', 'nARE', 'are','and','Unknown', 'H4','EAD','length','VISAS', 'dtype','startups'}
    #stopwords = stopwords.union(more_stopwords)
    #Generate the word cloud
    wordcloud = WordCloud(background_color='black',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=800, 
                    height=400,
                    mask = mask)
    wordcloud.generate(str(text))
    #set the plot parameters
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  
    
#ngram function
def ngram_extractor(text, n_gram):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

# Function to generate a dataframe with n_gram and top max_row frequencies
def generate_ngrams(df, n_gram, max_row):
    temp_dict = defaultdict(int)
    for question in df:
        for word in ngram_extractor(question, n_gram):
            temp_dict[word] += 1
    temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(max_row)
    temp_df.columns = ["word", "wordcount"]
    return temp_df

#Function to construct side by side comparison plots
def comparison_plot(df_1,df_2,col_1,col_2, space):
    fig, ax = plt.subplots(1, 2, figsize=(20,10))
    
    sns.barplot(x=col_2, y=col_1, data=df_1, ax=ax[0], color="royalblue")
    sns.barplot(x=col_2, y=col_1, data=df_2, ax=ax[1], color="royalblue")

    ax[0].set_xlabel('Word count', size=14)
    ax[0].set_ylabel('Words', size=14)
    ax[0].set_title('Top 20 Bi-grams in Descriptions', size=18)

    ax[1].set_xlabel('Word count', size=14)
    ax[1].set_ylabel('Words', size=14)
    ax[1].set_title('Top 20 Tri-grams in Descriptions', size=18)

    fig.subplots_adjust(wspace=space)
    
    plt.show()


# In[ ]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

#phrase = "Here is an example sentence to demonstrate the removal of stopwords"

#words = word_tokenize(phrase)

stripped_phrase = []
stripped_sentence = []
for sentence in sentences:
    for word in sentence:
        if word not in stop_words:
            stripped_phrase.append(word)
    stripped_sentence.append(stripped_phrase)

des_1 = " ".join(stripped_phrase)


# In[ ]:


plot_wordcloud(des_1, title="Word Cloud of Data Analyst Description")

