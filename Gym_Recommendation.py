

import requests
from bs4 import BeautifulSoup
import pandas as pd

from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
#https://www.crummy.com/software/BeautifulSoup/bs4/doc/



class reconSys():
    url = 'https://en.wikipedia.org/wiki/List_of_weight_training_exercises'
    
    
    def scrapePage(self):
        resp = requests.get(self.url)
        #Web scrapping library called beautifulsoup
        soup = BeautifulSoup(resp.text,'lxml')
        table = soup.find_all('table','wikitable')[0]
        df = pd.read_html(str(table))
        df = pd.concat(df)
        return df
    
    def prep(self):
        df = self.scrapePage()
        df = df.fillna(0).replace(['Some','Yes'],1)
        df['Exercise'] = df['Exercise'].str.title()
        return df
        
    
    def exerciseRecommend(self,exercise,no_exercises):
        df = self.prep()
        indices = pd.Series(df.index, index = df['Exercise'])
        df2 = df.iloc[:,df.columns != 'Exercise']
        cosine_sim = cosine_similarity(df2,df2)
        
        idx = indices[exercise]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = filter(lambda i:i[0] != idx, sim_scores)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 5 most similar exercises
    
        sim_scores = sim_scores[0:no_exercises]

    # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        print('Top {} Exercises based on {}'.format(no_exercises, exercise))
    # Return the top most similar exercises
        return df['Exercise'].iloc[movie_indices]



