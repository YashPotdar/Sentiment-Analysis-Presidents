#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis on American Presidents
# ## Yash Potdar
# ### Data Science UCSD '23
# 
# * Question: How does sentiment analysis perceive Franklin D. Roosevelt, Andrew Jackson, and Donald J. Trump's biographies on Wikipedia?
# * Background: Throughout American history, there have been presidents that have been widely regarded and praised, loathed by the public. However, there have also been presidents who have controversial images due to a mixture of good and bad actions. Although we, as humans, can form our own opinions about these presidents, can sentiment analysis using Vader Lexicon confirm our opinions?
#   * Franklin Delano Roosevelt, the 32nd President, has generally been praised due to his progressive New Deal policies that uplifted the country out of the Great Depression. He was the only president elected for four terms.
#   * Andrew Jackson, the 7th President, has a controversial, mixed image. On one hand, Jackson strongly supported slavery and demanded the forced removal of Native Americans on the Trail of Tears, displacing hundreds of thousands of Native Americans. On the other hand, he was known as a common man and destroyed the Second Bank of the US, which would protect individual liberties.
#   * Donald Trump, the 45th President, has a negative image due to his reversals of many progressive actions of prior presidents, and his general misogynistic and racist behavior.
# * **Part 1**: Determine the Average Compound Score for each President's Wiki Page
# * **Part 2**: Visualizing the Most Extreme Words for the Presidents
# 

# ## Part 1: Average Compound Sentiment Score for each President

# ### Step 1: Import Relevant Packages

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests
import re
from urllib.parse import urlparse
import urllib.robotparser
from bs4 import BeautifulSoup
from wordcloud import WordCloud, STOPWORDS

# This code checks the robots.txt file
def canFetch(url):

    parsed_uri = urlparse(url)
    domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)

    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(domain + "/robots.txt")
    try:
        rp.read()
        canFetchBool = rp.can_fetch("*", url)
    except:
        canFetchBool = None
    
    return canFetchBool


# ### Step 2: Permissions for Web Scraping
# Before I can retrieve text from the three Wikipedia pages, I must first send requests to the three Wikipedia websites, which are represented in the `sites` list. The package I am using to communicate with the websites is `requests`, from which I invoke the `get` function on each site. This gives the HTTP status code. A `200` response returned from the function call means the page was found and the request succeeded. After checking the HTTP status code for each website, I also used the `canFetch` method, which lets me know whether there is permission to use a robot to access the site. All the sites yielded True after calling this function, so web scraping was permitted
# is a green light, and means that web scraping is permitted.

# In[2]:


sites = ["https://en.wikipedia.org/wiki/Franklin_D._Roosevelt", "https://en.wikipedia.org/wiki/Andrew_Jackson", 
         "https://en.wikipedia.org/wiki/Donald_Trump"]
for site in sites:
    print('* Site: {0}\n\tHTTP Response: {1}\n\tWeb Scrape Permitted: {2}'          .format(site, requests.get(site), canFetch(site)))


# ### Step 3: Retrieving Data from the 3 Wikipedia pages
# Next, I accessed each president's corresponding URL from the `sites` list from Step 1. For example, FDR would correspond to index 0 and Trump would correspond to index 2. Then I retrieved the unicode response from each wiki page using the text property of the response from the GET HTTP method. I stored these in each president's `urlText` string.
# FDR's unicode text from his Wiki is below as an example. This is nowhere near the required form for sentiment analysis, so further cleaning is required. 

# In[3]:


def get_url_text(url_str):
    return requests.get(url_str, timeout = 5).text

# FDR: Franklin Delano Roosevelt
fdr_url = sites[0]
fdr_urlText = get_url_text(fdr_url)

# AJ: Andrew Jackson
aj_url = sites[1]
aj_urlText = get_url_text(aj_url)

# DJT: Donald Trump
djt_url = sites[2]
djt_urlText = get_url_text(djt_url)

#fdr_urlText


# ### Step 4: Cleaning Text Using BeautifulSoup
# Next, to clean the HTML and convert it into a more readable form, I used the BeautifulSoup package. For each president, I used the HTML parser in BeautifulSoup to extract the page's content. Finally, I filtered all irrelevant HTML tags by using the `find_all` method of `page_content`, a BeautifulSoup object. By focusing on only the paragraphs, we are one step closer to sentiment analysis. To verify that we indeed have paragraphs, the nested for loop prints all the paragraphs on Trump's Wiki page.

# In[4]:


def get_page_paragraphs(url_text_str):
    soup = BeautifulSoup(url_text_str, 'html.parser')
    return soup.find_all("p")

# FDR: Franklin Delano Roosevelt
fdr_paragraphs = get_page_paragraphs(fdr_urlText)

# AJ: Andrew Jackson
aj_paragraphs = get_page_paragraphs(aj_urlText)

# DJT: Donald Trump
djt_paragraphs = get_page_paragraphs(djt_urlText)

#for paragraph in djt_paragraphs:
#    print(paragraph.text)


# ### Step 5: Create List with All Cleaned Sentences from Wiki
# Next, for each president, I made a list that would contain all the paragraphs. Then, for every paragraph, I used Regular Expressions to remove items from brackets and parentheses. This is due to the fact that Wikipedia has references in brackets and parentheses which are irrelevant to my analysis. Next, I joined all the paragraphs in the list into one string so they could be easier cleaned. I then removed commas and quotation marks since they would be attached to words. I split the text with a period delimiter to have a list with all the sentences in the Wiki page. Finally, I used a list comprehension to filter out "sentences" that were not actually sentences because they were acronyms split by the function call (ex: U.S. would be split into 2 elements in `all_sentences_fdr`).

# In[5]:


regex_brackets = "\[.*\]|\s-\s.*"
regex_parentheses = "\(.*\)|\s-\s.*"

def get_all_text_cleaned(paragraphs):
    lst_paragraphs = []
    for par in paragraphs:
        lst_paragraphs.append(par.text)
    for item in range(len(lst_paragraphs)):
        lst_paragraphs[item] = lst_paragraphs[item].strip('\n').lower().replace(',', '').replace('"', '')
        lst_paragraphs[item] = re.sub(regex_brackets, "", lst_paragraphs[item])
        lst_paragraphs[item] = re.sub(regex_parentheses, "", lst_paragraphs[item])
    all_cleaned_text = " ".join(lst_paragraphs).strip()
    return all_cleaned_text

def get_all_sentences(all_text_str):
    sentences = all_text_str.split('.')
    sentences = [s for s in sentences if len(s.split()) > 1]
    return sentences
    

# FDR: Franklin Delano Roosevelt
all_text_fdr = get_all_text_cleaned(fdr_paragraphs)
all_sentences_fdr = get_all_sentences(all_text_fdr)

# AJ: Andrew Jackson
all_text_aj = get_all_text_cleaned(aj_paragraphs)
all_sentences_aj = get_all_sentences(all_text_aj)

# DJT: Donald Trump
all_text_djt = get_all_text_cleaned(djt_paragraphs)
all_sentences_djt = get_all_sentences(all_text_djt)

all_sentences_aj[:15]


# ### Step 6: Sentiment Analysis on Every Sentence
# Next, I used the Vader `SentimentIntensityAnalyzer` package to analyze the sentiments of every sentence in each president's Wiki. I created lists for each president to store the results of the sentiment calculation. 

# In[6]:


sid = SentimentIntensityAnalyzer()

def get_sentiment_lst(word_or_sentence_lst):
    """
    This function can be used to find the sentiments of either (1) a list of all the sentences on the page \
    or (2) a list of all the words on the page
    """
    sentiments = []
    for phrase in word_or_sentence_lst:
        phrase_Sentiment = sid.polarity_scores(phrase)
        phrase_Sentiment['text'] = phrase
        sentiments.append(phrase_Sentiment)
    return sentiments


# FDR: Franklin Delano Roosevelt
fdr_Sentiments = get_sentiment_lst(all_sentences_fdr)

# AJ: Andrew Jackson
aj_Sentiments = get_sentiment_lst(all_sentences_aj)

# DJT: Donald Trump
djt_Sentiments = get_sentiment_lst(all_sentences_djt)

djt_Sentiments[:5]


# ### Step 7: DataFrame for Each President
# Next, I made a dataframe for each president, which would hold the results from the sentiment analysis on each sentence. The first 15 rows of the sentiment DataFrame for Jackson.

# In[7]:


fdrSentimentDf = pd.DataFrame(fdr_Sentiments)
ajSentimentDf = pd.DataFrame(aj_Sentiments)
djtSentimentDf = pd.DataFrame(djt_Sentiments)

ajSentimentDf.head(10)


# ### Step 8: Combined DataFrame for Average Compound Sentiment Scores
# Next, I made a list that contained the averages of the `compound` column of each of the DataFrames from Step 8. Then I made a single DataFrame that contained the average compound sentiment scores for each president's Wiki.

# In[8]:


avg_compound = [np.mean(pres_df['compound']) for pres_df in [fdrSentimentDf, ajSentimentDf, djtSentimentDf]]
df_compound = pd.DataFrame({'President':['FDR', 'Jackson', 'Trump'], 'Avg Compound':avg_compound})
df_compound


# ### Step 9: Visualizing Average Compound Sentiment Scores for Each President
# To better visualize the results, I made a bar graph of the Average Compound Sentiment Scores. 

# In[9]:


ax = df_compound.plot(x ='President', y ='Avg Compound', kind = 'bar', legend = False,     figsize = [10, 30/4], fontsize = 13)
ax.set_title('Sentiment Analysis Scores From Each President\'s Wiki',fontsize= 19)
ax.set_xlabel('President', fontsize = 15)
ax.set_ylabel('Average Compound Score', fontsize = 17)


# ### Conclusion for Part 1: 
# The results for Part 1 were not what I had expected, and it suggests that Vader lexicon was not able to correctly reflect the public's perception of the Presidents. I would expect FDR's compound score to be much higher than Trump's and Trump's to be a high negative percent. Jackson's was the only score that I expected to be around neutral since he had such a controversial legacy. The results may be due to the fact that this was from Wikipedia, so the sentences may be written in a more objective manner, which draws the sentiment closer to neutral. 

# ## Part 2: Visualizing the Most Polarizing Words 

# ## Steps 1-5 from Part 1
# A key difference in the sentiment analysis from Part 1 and Part 2 is that in Part 1, analysis was done on sentences as a whole. In Part 2, since the goal is to make a word cloud of the most polarizing words, analysis will be done on individual words. 
# For my analysis, I will be using `all_text_fdr`, `all_text_aj`, and `all_text_djt` from Step 5 of Part 1. These three variables represent strings that contain all the cleaned text from the Wikipedia page. 

# In[10]:


def get_individual_words(all_text_str):
    return all_text_str.replace('.','').split()

all_words_fdr = get_individual_words(all_text_fdr)
all_words_aj = get_individual_words(all_text_aj)
all_words_djt = get_individual_words(all_text_djt)

all_words_fdr[:15]


# ## Step 6: Sentiment Analysis on All the Text
# In this step, I did sentiment analysis on every word in each president's Wiki page. An initial thought that came to mind was to filter out duplicates since it would decrease the runtime of the code. However, this would entirely defeat the purpose of the end goal of making a wordcloud, since a wordcloud uses frequencies of words to determine scaling. Moreover, I was debating whether to filter out stop words, but decided against it since I could use the STOPWORD package in WordCloud which would be more precise than determining if a word is a stop word just on the basis of length. Similar to Step 6 of Part 1, I created lists for each president to store the results of the sentiment calculation.

# In[11]:


# FDR: Franklin Delano Roosevelt
fdr_Sentiments2 = get_sentiment_lst(all_words_fdr)

# AJ: Andrew Jackson
aj_Sentiments2 = get_sentiment_lst(all_words_aj)

# DJT: Donald Trump
djt_Sentiments2 = get_sentiment_lst(all_words_djt)

djt_Sentiments2[:15]


# ## Step 7: Dataframe
# Similar to Step 7 of Part 1, I made a dataframe for each president, which would hold the results from the sentiment analysis on each word. The first 15 rows of the sentiment DataFrame for Trump.

# In[12]:


fdrSentimentDf2 = pd.DataFrame(fdr_Sentiments2)
ajSentimentDf2 = pd.DataFrame(aj_Sentiments2)
djtSentimentDf2 = pd.DataFrame(djt_Sentiments2)

djtSentimentDf2.head(15)


# ## Step 8: Filtering the Most Positive or Negative Words
# In this step, I decided to find the most positive words for FDR, most negative words for Trump, and both for Jackson. I first sorted each DataFrame by the `compound` column since this was a better indicator of the positivity/negativity of the word than the `pos` or `neg` columns. I decided to include only the 100 most extreme words since I did not want to reach any neutral words, and I wanted there to be repetition between the words so it could be seen in the wordcloud. I then joined the words in the list into a string, which would then be used to generate a wordcloud.

# In[13]:


def find_most_polarizing_words(sentiment_df):
    most_positive_lst = list(sentiment_df.sort_values(by = ['compound'], ascending = False)[:100]['text'])
    most_positive_str = " ".join(most_positive_lst)
    most_negative_lst = list(sentiment_df.sort_values(by = ['compound'], ascending = True)[:100]['text'])
    most_negative_str = " ".join(most_negative_lst)
    yield most_positive_str
    yield most_negative_str

    
# FDR: Franklin Delano Roosevelt
fdr_generator = find_most_polarizing_words(fdrSentimentDf2)
most_pos_fdr = next(fdr_generator)
most_neg_fdr = next(fdr_generator)

# AJ: Andrew Jackson
aj_generator = find_most_polarizing_words(ajSentimentDf2)
most_pos_aj = next(aj_generator)
most_neg_aj = next(aj_generator)

# DJT: Donald Trump
djt_generator = find_most_polarizing_words(djtSentimentDf2)
most_pos_djt = next(djt_generator)
most_neg_djt = next(djt_generator)

most_neg_djt


# ## Step 8: Visualizations of Most Frequent Extreme Words
# Here, I used a WordCloud object to display the most extreme words. 

# In[14]:


cloud = WordCloud(background_color = "white", max_words = 25, collocations = False, stopwords = set(STOPWORDS))

def make_wordCloud(most_polarizing_words_str):
    cloud.generate(most_polarizing_words_str)
    plt.figure(figsize = (9, 9), facecolor = None) 
    plt.imshow(cloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    

# FDR: Franklin Delano Roosevelt
make_wordCloud(most_pos_fdr)
plt.title('Most Frequent Positive Words About Franklin D. Roosevelt')

make_wordCloud(most_neg_fdr)
plt.title('Most Frequent Negative Words About Franklin D. Roosevelt')

# AJ: Andrew Jackson
make_wordCloud(most_pos_aj)
plt.title('Most Frequent Positive Words About Andrew Jackson')

make_wordCloud(most_neg_aj)
plt.title('Most Frequent Negative Words About Andrew Jackson')

# DJT: Donald Trump
make_wordCloud(most_pos_djt)
plt.title('Most Frequent Positive Words About Donald J. Trump')

make_wordCloud(most_neg_djt)
plt.title('Most Frequent Negative Words About Donald J. Trump')


plt.show()


# ### Conclusion for Part 2: 
# From the wordclouds, it's apparent that the words included are the most extreme, and suggest that Vader lexicon was a good way to gauge how positive or negative a word was, and its compound score was a good estimator. 

# # Further Study:
# In the future, I will be doing further experimentation with text analysis on Presidents or candidates during elections over time by comparing text in their speeches. This could very well demonstrate how the field of politics has changed over the past decades. 

# In[ ]:




