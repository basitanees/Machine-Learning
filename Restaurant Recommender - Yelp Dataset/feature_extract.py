# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 01:55:10 2019

@author: basit
"""
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import ast
atts = business['attributes']

def get_atts(att,feature):
    out = np.zeros((57402,1))
    num = -1
    for j in att:
        x = att.loc[num+1]
        num = num + 1
        for i in x:
            if i == feature:
                if x.get(i) == 'True':
                    out[num,0] = 1
                    break
                else:
#                    out[num,0] = 0
                    break
#            out[num,0] = np.nan
    return out
        
hasTV = get_atts(atts,'HasTV')
reserve = get_atts(atts,'RestaurantsReservations')
caters = get_atts(atts,'Caters')
seating = get_atts(atts,'OutdoorSeating')
delivery = get_atts(atts,'RestaurantsDelivery')
gfk = get_atts(atts,'GoodForKids')
gfb = get_atts(atts,'GoodForBabies')

def get_atts_1(att,feature,feature_in):
    out = np.zeros((57402,1))
    num = -1
    for j in att:
        x = att.loc[num+1]
        num = num + 1
        for i in x:
            if i == feature:
                new = ast.literal_eval(x.get(i))
                if isinstance(new, dict):
                    for k in new:
                        if k == feature_in:
                            if new.get(k) == True:
                                out[num,0] = 1
                                break
                            else:
                                out[num,0] = 0
#                                print(0)
                                break
    return out

gfm = get_atts_1(atts,'GoodForMeal','lunch')
gfd = get_atts_1(atts,'GoodForMeal','dinner')
parking = get_atts_1(atts,'BusinessParking','lot')
ambience = get_atts_1(atts,'Ambience','casual')

features = np.hstack((hasTV,reserve,caters,seating,delivery,gfk,gfb,gfm,gfd,parking,ambience))
features_new = np.hstack((vc,features))
features_new1 = pd.DataFrame(features_new)
avg_stars1 = business['stars'].values
avg_stars = avg_stars1/5
avg_stars = pd.DataFrame(avg_stars).rename(columns = {0:74})
features_new2 = pd.concat([business[['business_id']],features_new1,avg_stars],axis = 1)
features_new2.to_json('C:/3rd Year/Sixth Semester/EEE 485/Project Datasets/Yelp 1/yelp_academic_dataset_features_all_2.json')
