import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st

from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity



def data_clean():
    df = pd.read_json('Data/fullex.json')

    #Remove text before colon in each column
    df['Muscle Group'] = df['Muscle Group'].apply(lambda x: re.sub(r'^.*?:','',x)).str.lstrip()
    df['Exercise Type'] = df['Exercise Type'].apply(lambda x: re.sub(r'^.*?:','',x)).str.lstrip()
    df['Equipment'] = df['Equipment'].apply(lambda x: re.sub(r'^.*?:','',x)).str.lstrip()
    df = df.rename(columns={'title':'Exercise'})
    df = df.sort_values(by='Exercise').reset_index(drop=True)

    #Table sort and drop dups
    df2 = df.drop_duplicates().reset_index(drop=True)
    df2 = df2.sort_values(by='Exercise').reset_index(drop=True)

    #Remove random white spaces (leading, trailing or double whitespaces)
    df2['Exercise']= df2['Exercise'].apply(lambda x: " ".join(x.split()))
    df2['Muscle Group'] = df2['Muscle Group'].apply(lambda x: " ".join(x.split()))
    df2['Exercise Type']= df2['Exercise Type'].apply(lambda x: " ".join(x.split()))
    df2['Equipment']= df2['Equipment'].apply(lambda x: " ".join(x.split()))


    #Cleaning Muscle Group 
    ham = ['Romanian', 'Hamstring', 'Leg Curl']
    quad = ['Quad','Leg Extension', 'Squat', 'Lunge', 'Leg Press']

    df2['Muscle Group'] = df2.apply(lambda x: ('Hamstring' if any(i in x['Exercise'] for i in ham) else x['Muscle Group']) if x['Muscle Group'] == 'Upper Legs' else x['Muscle Group'],axis = 1)
    df2['Muscle Group'] = df2.apply(lambda x: ('Quads' if any(i in x['Exercise'] for i in quad) else x['Muscle Group']) if x['Muscle Group'] == 'Upper Legs' else x['Muscle Group'],axis = 1)
    df2['Muscle Group'] = df2.apply(lambda x: "Calf" if x['Muscle Group'] == 'Lower Legs' else x['Muscle Group'],axis = 1)
    df2['Muscle Group'] = df2.apply(lambda x: "Glutes" if "Hip" in x['Muscle Group'] else x['Muscle Group'],axis = 1)

    #Create a Push, Pull, Legs map
    push = ['Shoulders', 'Chest', 'Triceps']
    pull = ['Back', 'Forearm', 'Biceps']
    legs = ['Hamstring', 'Abs', 'Cardio', 'Quads', 'Upper Legs', 'Lower Legs', 'Glutes', 'Calf']

    df2['PPL Map'] = None
    df2['PPL Map'] = df2.apply(lambda x: 'Push' if any(i in x['Muscle Group'] for i in push) else x['PPL Map'] ,axis = 1)
    df2['PPL Map'] = df2.apply(lambda x: 'Pull' if any(i in x['Muscle Group'] for i in pull) else x['PPL Map'] ,axis = 1)
    df2['PPL Map'] = df2.apply(lambda x: 'Legs' if any(i in x['Muscle Group'] for i in legs) else x['PPL Map'] ,axis = 1)

    return df, df2


def prep(df):
    #df = self.filter()
    df = df.groupby('Exercise').agg({'Exercise Type':', '.join, 
                            'Muscle Group': ', '.join,
                            'PPL Map': ', '.join, 
                            'Equipment':', '.join}).reset_index()
    
    # Get all unique values in columns
    # Tokenise each word in the columns 
    df['Exercise Type'] = df['Exercise Type'].apply(lambda x: ', '.join(list(dict.fromkeys(x.split(', '))))).str.split(", ")
    df['Muscle Group'] = df['Muscle Group'].apply(lambda x: ', '.join(list(dict.fromkeys(x.split(', '))))).str.split(", ")
    df['PPL Map'] = df['PPL Map'].apply(lambda x: ', '.join(list(dict.fromkeys(x.split(', '))))).str.split(", ")
    df['Equipment'] = df['Equipment'].apply(lambda x: ', '.join(list(dict.fromkeys(x.split(', '))))).str.split(", ")

    ####### Encoding ########


    mg_encode = pd.concat(
        [
            df.explode('Muscle Group')
            #Transform each element of a list-like to a row, replicating index values.
            .pivot_table(index='Exercise', columns='Muscle Group', aggfunc="size", fill_value=0)
            #pivots table with column route as its focus (columns)
            .add_prefix("Muscle_"),
            #df.set_index("id").ticket_price,
        ],
        axis=1,
    )


    et_encode = pd.concat(
        [
            df.explode('Exercise Type')
            #Transform each element of a list-like to a row, replicating index values.
            .pivot_table(index='Exercise', columns='Exercise Type', aggfunc="size", fill_value=0)
            #pivots table with column route as its focus (columns)
            .add_prefix("ExType_"),
            #df.set_index("id").ticket_price,
        ],
        axis=1,
    )


    ppl_encode = pd.concat(
        [
            df.explode('PPL Map')
            #Transform each element of a list-like to a row, replicating index values.
            .pivot_table(index='Exercise', columns='PPL Map', aggfunc="size", fill_value=0)
            #pivots table with column route as its focus (columns)
            .add_prefix("PPL_"),
            #df.set_index("id").ticket_price,
        ],
        axis=1,
    )


    equip_encode = pd.concat(
        [
            df.explode('Equipment')
            #Transform each element of a list-like to a row, replicating index values.
            .pivot_table(index='Exercise', columns='Equipment', aggfunc="size", fill_value=0)
            #pivots table with column route as its focus (columns)
            .add_prefix("Equip_"),
            #df.set_index("id").ticket_price,
        ],
        axis=1,
    )

    # merge all encoding dataframes
    df2 = df.merge(mg_encode,on='Exercise').merge(et_encode,on='Exercise').merge(ppl_encode,on='Exercise').merge(equip_encode,on='Exercise')
    # Create a list for all columns except the ones stated below, as they are encoded
    cols = list(filter(lambda col: not any(i in col for i in ['Muscle Group', 'Exercise Type', 'Equipment', 'PPL Map']), df2.columns))
    df2 = df2[cols]

    return df2



def recommend(df, exercise, no_exercises):

    #Create df to compare against each other
    df2 = df.iloc[:,df.columns != 'Exercise']

    indices = pd.Series(df.index, index = df['Exercise'])

    # Compute the cosine similarity matrix
    
    cosine_sim = cosine_similarity(df2,df2)

    # Get index for chosen exercise
    idx = indices[exercise]

    # Get the scores with their index [(idx,score), (idx, score),.., (idx, score)]
    sim_scores = list(enumerate(cosine_sim[idx]))

    #Filter list to exclude chosen exercise
    sim_scores = list(filter(lambda i:i[0] != idx, sim_scores))

    #sort list by their scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    fil_scores = sim_scores[0:no_exercises]

    ex_indices = [i[0] for i in fil_scores]

    print('Top {} Exercises based on {}'.format(no_exercises, exercise))
    # Return the top most similar exercises
    return df['Exercise'].iloc[ex_indices]




####### STREAMLIT APP BUILD #########


#Create Sidebar - all attributes of sidebar will start with st.sidebar
st.sidebar.header('User Input Features')

exerciseData = data_clean()[1]

#SideBar - PPL Selection
selectedGroup = st.sidebar.selectbox('PPL',exerciseData['PPL Map'].unique().tolist())

#SideBar - Exercise Type Selection

container = st.sidebar.container()
types_all = st.sidebar.checkbox("Select all Exercise Types")

type_selectors = list(dict.fromkeys([i.split(',')[0] for i in exerciseData['Exercise Type'].unique().tolist()]))

if types_all:
    selectedType = container.multiselect("Select one or more Exercise Types:",
         type_selectors,type_selectors)
else:
    selectedType =  container.multiselect("Select one or more Exercise Types:",
        type_selectors)


#SideBar - Equipment Selection

container = st.sidebar.container()
Equip_all = st.sidebar.checkbox("Select all Equips")

equip_selectors = list(dict.fromkeys([i.split(',')[0] for i in exerciseData['Equipment'].unique().tolist()]))

if Equip_all:
    selectedEquip = container.multiselect("Select one or more Equipment Types:",
         equip_selectors,equip_selectors)
else:
    selectedEquip =  container.multiselect("Select one or more Equipment Types:",
        equip_selectors)

filterData = exerciseData[(exerciseData['PPL Map'].isin(selectedGroup.split())) & (exerciseData['Exercise Type'].str.contains('|'.join(selectedType))) & (exerciseData['Equipment'].str.contains('|'.join(selectedEquip)))]


# Sidebar - Exercise Select
selectedEx = st.sidebar.selectbox('Select your exercise',filterData['Exercise'].unique().tolist())

#Sidebar - Number of exercises
selectedNum = st.sidebar.slider('No of Exercises',5,20,20)

#Recommendation system
engine = recommend(prep((filterData)), selectedEx,selectedNum)

######## Page Set up ########
st.title('Exercise Recommendation System')

st.markdown("""

This app aims to show similar exercises to one selected.
* **Key Python libraries:** pandas, sklearn
* **Data source:** webscraped from Jefit
""")

st.dataframe(filterData.reset_index(drop=True))

st.write('Top {} exercises based on {}'.format(selectedNum,selectedEx))
st.table(engine.reset_index(drop=True))
