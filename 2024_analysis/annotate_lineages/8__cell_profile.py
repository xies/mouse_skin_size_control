#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 14:55:03 2025

@author: xies
"""


#%% Go backwards from the delamination frame

delam = df[df['Delaminate next frame','Meta']]

Niter = 100
birth = df[df['Birth frame','Meta']]

feature_names = features.index
importances = pd.DataFrame(columns=feature_names)
avg_confusion = np.zeros((Niter,2,2))

for i in tqdm(range(Niter)):
    
    df_ = get_balanced_df_by_category(delam, delam['Will differentiate','Meta'].values)
    
    y = df_['Will differentiate','Meta']
    X = preprocessing.scale(df_.xs('Measurement',level=1,axis=1))
    
    X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y)
    
    forest = ensemble.RandomForestClassifier().fit(X_train,y_train)
    
    y_pred = forest.predict(X_test)
    avg_confusion[i,...] = metrics.confusion_matrix(y_test,y_pred)/len(y_test)
    
    importances.loc[i] = forest.feature_importances_

sb.heatmap(avg_confusion.mean(axis=0),annot=True)
print(importances.mean().sort_values().tail(20))





