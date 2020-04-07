import pandas as pd
import numpy as np
import csv
from hunspell import Hunspell
import re

spell = Hunspell()

# add prompt related data into spell checker
print('---goldstandard---')
for i in range(1,11):
    gold_df_train = pd.read_csv('data/asap/gold/'+str(i)+'_train_gold.tsv',
                     encoding='utf-8', sep='\t', header=0, quoting=csv.QUOTE_NONE,
                     names=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'],
                     dtype={'Id': str, 'EssaySet': str, 'essay_score1': np.int32,'essay_score2': np.int32, 'EssayText': str})
    gold_df_test = pd.read_csv('data/asap/gold/'+str(i)+'_test_gold.tsv',
                     encoding='utf-8', sep='\t', header=0, quoting=csv.QUOTE_NONE,
                     names=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'],
                     dtype={'Id': str, 'EssaySet': str, 'essay_score1': np.int32,'essay_score2': np.int32, 'EssayText': str})
    gold_df = pd.concat([gold_df_train,gold_df_test])
    words = " ".join(gold_df.EssayText).split()
    unique_words = list(dict.fromkeys(words))
    for w in unique_words:
        if w in ['\uff1f','\u2018','\u2019']:
            continue
        spell.add(w)

# On adversarials
for i in range(1,11):
    print('---prompt ' + str(i) + '---')
    for r in range(14,15):
        print('threshold:'+str(r))
        print('---Adver---')
        adver_TPR = []
        df = pd.read_csv('adversarial/shallow/adversarial_prompt_'+str(i)+'.txt',
                         encoding='utf-8', sep='\t', header=0, quoting=csv.QUOTE_NONE,
                         names=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'],
                         dtype={'Id': str, 'EssaySet': str, 'essay_score1': np.int32,'essay_score2': np.int32, 'EssayText': str})
        df['error rate'] = np.nan
        for index, row in df.iterrows():
            if type(row['EssayText']) is float:
                df.at[index, 'error rate'] = 100
                continue
            token_list = row['EssayText'].split()
            count_error = 0
            for t in token_list:
                if spell.spell(t) is False:
                    count_error+=1
            if count_error!= 0:
                df.at[index, 'error rate'] = count_error/len(token_list)*100
            else:
                df.at[index, 'error rate'] =0
        random_char = df.iloc[:1000]
        random_word = df.iloc[1000:2000]
        brown_char_ngram = df.iloc[2000:7000]
        brown_word_ngram = df.iloc[7000:12000]
        asap_char_ngram = df.iloc[12000:17000]
        asap_word_ngram = df.iloc[17000:22000]
        content_burst = df.iloc[22000:23000]
        shuffle = df.iloc[23000:24000]
        gpt_2 = df.iloc[24000:25001]
        sub_dfs = [random_char,random_word,brown_char_ngram,brown_word_ngram,asap_char_ngram,asap_word_ngram,content_burst,shuffle,gpt_2]
        for sub_df in sub_dfs:
            #print(sub_df)
            #print(len(sub_df[sub_df['error rate'] > r])/1000)
            adver_TPR.append(len(sub_df[sub_df['error rate'] > r])/1000)
        adver_TPR[2] = adver_TPR[2] / 5
        adver_TPR[3] = adver_TPR[3] / 5
        adver_TPR[4] = adver_TPR[4] / 5
        adver_TPR[5] = adver_TPR[5] / 5
        for item in adver_TPR:
            print(item)
        print('---Origin---')
        origin_TNR = 0
        df = pd.read_csv('data/asap/test_public_repaired.txt',
                         encoding='utf-8', sep='\t', header=0, quoting=csv.QUOTE_NONE,
                         names=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'],
                         dtype={'Id': str, 'EssaySet': str, 'essay_score1': np.int32,'essay_score2': np.int32, 'EssayText': str})
        df = df.loc[df['EssaySet']==str(i)]
        df['error rate'] = np.nan
        for index, row in df.iterrows():
            if type(row['EssayText']) is float:
                df.at[index, 'error rate'] = 100
                continue
            token_list = row['EssayText'].split()
            count_error = 0
            for t in token_list:
                if spell.spell(re.sub(r'[^\w\s]','',t)) is False:
                    count_error += 1
            if count_error != 0:
                df.at[index, 'error rate'] = count_error / len(token_list) * 100
            else:
                df.at[index, 'error rate'] = 0

        print(len(df[df['error rate'] < r])/len(df))
        origin_TNR = len(df[df['error rate'] < r])/len(df)
        print('maximal average =',(np.mean(adver_TPR)+origin_TNR)/2)
