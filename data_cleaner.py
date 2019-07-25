# -*- coding: utf-8 -*-
# Author: chen
# Created at: 6/12/19 10:20 AM

import data_helper as dh
import pandas as pd

def generate_text(dataframe):
    """Return combined texts and titles"""
    body = nonetostr(list(dataframe.text))
    titles = nonetostr(list(dataframe.title))
    
    for i,x in enumerate(titles):
        if x == None:
            titles[i] = ''
    
    texts = [body[i] + titles[i] for i, x in enumerate(titles)]
    
    return texts


def nonetostr(text_list):
    """Change all none type elements to str"""
    for i,x in enumerate(text_list):
        if x == None:
            text_list[i] = ''
    
    return text_list


def searchkw(text_list, kw_lists):
    """Search for targeted elements in the list based on the lw_lists"""
    mentions_list = []
    for i, x in enumerate(text_list):
        if any(n in x.lower() for n in kw_lists):
            mentions_list.append(i)
            
    return mentions_list


main_stream = dh.load_json("data/Avengers/avengers.json")

texts = generate_text(main_stream)

mentions = searchkw(texts, ['lbj', 'lebron', 'lebron james'])

# Get all posts related to James
james_dataframe = main_stream.iloc[mentions]
james_main_id = list(james_dataframe.sub_id)

# Generate James related 
james_reply_stream = pd.DataFrame([])
for i in range(8):
    df = dh.load_json('data/NBA_1904/reply_stream_{}.json'.format(i+1))
    for i in range(len(df)):
        if df.iloc[i].link_id in james_main_id:
            james_reply_stream = james_reply_stream.append(df.iloc[i])
    
    print("{}, Done".format(i))
    









