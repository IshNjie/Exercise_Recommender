

import requests
from bs4 import BeautifulSoup
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
#https://www.crummy.com/software/BeautifulSoup/bs4/doc/



class reconSys():
    url = 'https://en.wikipedia.org/wiki/List_of_weight_training_exercises'
    #Set the url to retrieve the information
    
    def scrapePage(self):
        resp = requests.get(self.url)
        #Web scrapping library called beautifulsoup
        soup = BeautifulSoup(resp.text,'lxml')
        
        #Locate the table needed from the wiki page 
        table = soup.find_all('table','wikitable')[0]

        #Convert HTML text to df
        df = pd.read_html(str(table))[0]
        #df = pd.concat(df)
        return df
    
    def prep(self):

        #Get df from scrape function
        df = self.scrapePage()

        #Clean the data and replace text with binary values - treating N/A as 0
        df = df.fillna(0).replace(['Some','Yes'],1)
        df['Exercise'] = df['Exercise'].str.title()
        return df
        
    
    def exerciseRecommend(self,exercise,no_exercises):
        df = self.prep()

        #Get Index for each exercise
        indices = pd.Series(df.index, index = df['Exercise'])

        #Create df to perform pairwise similarity
        df2 = df.iloc[:,df.columns != 'Exercise']

        #Create matrix of similarity
        cosine_sim = cosine_similarity(df2,df2)
        
        #Get Index for specified Exercise
        idx = indices[exercise]

        #Get list of the cosine similarities of all exercises for specified exercise
        sim_scores = list(enumerate(cosine_sim[idx]))

        #Filter list of tuples to exclude index for specified exercise, as that will be 1
        sim_scores = filter(lambda i:i[0] != idx, sim_scores)

        #Sort tuples based on similarity measure, 2nd item in tuple
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the most similar exercises - filtered to amount specified
    
        sim_scores = sim_scores[0:no_exercises]

        # Get the list of exercies that are most similar
        movie_indices = [i[0] for i in sim_scores]

        print('Top {} Exercises based on {}'.format(no_exercises, exercise))
        # Return the top most similar exercises by filtering the starting dataframe
        return df['Exercise'].iloc[movie_indices]



