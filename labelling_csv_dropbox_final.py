#tool
import json
import pandas as pd
import numpy as np
import glob
import json
import re
import webbrowser, os
import datetime
import time

#ML
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline

#Setting paths for the data
document_paths = glob.glob("/tables/*.json")
csv_path = "df_labelling_final.csv"

#Accessing the data
df = pd.read_csv(csv_path, dtype={"type":str}, index_col=[0])
df = df.reset_index(drop=True)

#List of available classes
correct_classes = [
    "Bilanz", "Aktiva", "Passiva", "Gewinn und Verlust Rechnung", 
    "Kapitalflussrechnung", "Rückstellungen", "Forderungen", 
    "Verbindlichkeiten", "Anteilsbesitz", "Arbeitnehmer", 
    "Entwicklung des Anlagevermögens", "Vermögenswerte", "Umsatzerlöse",
    "Information", "not relevant"
    ]

def run_ML(df):
    """Runs the RandomForestClassifier with GridSearch over the datafram"""

    print("\n...Initiating the model training...")
    #Timing it
    start_time = time.time()

    #DF for training
    df_labeled = df.dropna(axis=0)

    df_labeled.content_stemmed
    df_labeled.type

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer()), #option3
        ('clf', RandomForestClassifier())
    ])

    param_grid = {
        'tfidf__max_df':  [0.9],
        'tfidf__min_df': [0.1],
        'clf__min_samples_leaf': [1, 2, 3], 
        'clf__n_estimators': [50, 100, 200]
    }

    grid_search = GridSearchCV(pipe, param_grid, scoring="accuracy", cv=5)
    grid_search.fit(df_labeled.content_stemmed, df_labeled.type)

    #Defining model with best parameters
    best_rf = grid_search.best_estimator_ 

    #Print best parameters
    print("\n...The training is complete!...\nThe best training parameters are {}".format(grid_search.best_params_))

    #Generate predictions
    predicted_labels = best_rf.predict(df.content_stemmed)

    #Generate probability
    max_prob = best_rf.predict_proba(df.content_stemmed)

    #Inputting predictions and max_prob
    for i, row in df.iterrows():
        df.at[i, 'prediction'] = predicted_labels[i]
        df.at[i, 'max_prob'] = max(max_prob[i])

    #Sort values by max_prob, reset index
    df = df.sort_values(by=['max_prob'], ascending=True)
    df = df.reset_index(drop=True)
    # print(df.max_prob.head()) #test check for correct sorting
    print("\n--- It took %s seconds to complete the training ---" % (time.time() - start_time))
    print('Total number of labled tables: ' +str(len(df_labeled)))
    return df
    

#Run the Model
df = run_ML(df)

number_of_reviewed_docs = []
#ITERATING THROUGH document_paths
for i, row in df.iterrows():
    if df['type'][i] not in correct_classes or pd.isnull(df['type'][i]):
        if (i != 0) and (i % 5 == 0):
            #rerun the model
            df = run_ML(df)
            #reset the index
            i = 0

            #Calculate the mean of max_prob to see if it improved
            mean_old = df.max_prob.mean()
            mean_prob_change = round((100*df.max_prob.mean()/mean_old)-100, 2)
            print('You teached me a lot, thanks :). I am now '+ str(improvement) + '% more (or less) confident on average about every classification. My mean confidence now is at '+ str(round(100*df.max_prob.mean(),2))+ ' %. Before it was at '+ str(round(100*mean_old,2))+ ' %')
        
        #opening relevant html file
        try:
            webbrowser.open('file://' + os.getcwd() + "/tables/{}.html".format(df['json'][i]))
        except:
            print("Problem with HTML! Please check")
            print("Here's the path: "+ str(os.getcwd()))
            
        #printing table text:
        print("\nLooking at document number " + str(i))
        print("Looking at TABLE_ID: " + str(df['id'][i]))
        print("Looking at DOCUMENT_ID: " + str(df['json'][i]))
        print("Here's the MAXIMUM PROBABILITY of this class: " + str(df['max_prob'][i]))
        print("Here's the PREDICTED CLASS: " + str(df['prediction'][i]))
        print("\nHere's the list of all classes:")
        print(correct_classes)
        print("\n\nTable contents: " + str(df.content[i]))
        print("\n\nLet's define the class!")
        #requesting label name
        new_label = input("\nWhich class is this table? Copy from the list above.\n>>> ")
        if new_label == 'end':
            df.to_csv('df_labelling_final.csv') #add ", index=False"?
            print("All changes saved, exiting the program")
            break
        elif new_label == 'skip':
            print("Alright, skipping to another example.")
            continue
        else:
            #need to check against a list of categories (maybe suggest a category?)
            if new_label in correct_classes:
                df.at[i, 'type'] = new_label #inserting new label
                df.at[i, 'label_added'] = datetime.datetime.now()
                df.to_csv('df_labelling_final.csv') #add ", index=False"?
                #printing new label:
                print("\nAWESOME! The new assigned label is: " + str(df['type'][i]))
                number_of_reviewed_docs.append(new_label)
            else:
                print("Incorrect label name! Opening another file.")
            #Checking next item
            print("\n############################################ \nHere's the next file: ")
        #run the ML every 5 classifications
        
        

print("Congrats, you've labelled {} documents!".format(str(len(number_of_reviewed_docs))))
labeled_count = round(df['type'].count() / len(df)*100, 2)
print("In total, you've labeled {}% of data.".format(labeled_count))