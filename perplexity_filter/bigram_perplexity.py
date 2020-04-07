import nltk
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from nltk.lm import Vocabulary
from nltk.corpus import brown
from nltk.util import ngrams
import csv
import pandas as pd
import numpy as np

# use brown as training data
tokenized_text = list(brown.sents())
n = 2
train_data = [nltk.bigrams(t, pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
words = [word for sent in tokenized_text for word in sent]
words.extend(["<s>", "</s>"])
padded_vocab = Vocabulary(words)
model = MLE(n)
model.fit(train_data, padded_vocab)

for p in range(1,11):
    # select test data in certain prompt
    test_df = pd.read_csv('data/asap/test_public_repaired.txt',
                           encoding='utf-8', sep='\t', header=0, quoting=csv.QUOTE_NONE,
                           names=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'],
                           dtype={'Id': str, 'EssaySet': str, 'essay_score1': np.int32, 'essay_score2': np.int32,
                                  'EssayText': str})
    prompt_df = test_df.loc[test_df['EssaySet'] == str(p)]
    test_sentences = prompt_df['EssayText'].tolist()
    tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in test_sentences]
    test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]

    perplexity_list = []
    for i, test in enumerate(test_data):
        #print("PP({0}):{1}".format(test_sentences[i], model.perplexity(test)))
        perplexity_list.append(model.perplexity(test))
    print('TNR on ASAP-SAS: ', (1-perplexity_list.count(float("inf"))/len(perplexity_list))*100)

    print('---Adver---')
    df = pd.read_csv('adversarial/shallow/adversarial_prompt_' + str(p) + '.txt',
                     encoding='utf-8', sep='\t', header=0, quoting=csv.QUOTE_NONE,
                     names=['Id', 'EssaySet', 'essay_score1', 'essay_score2', 'EssayText'],
                     dtype={'Id': str, 'EssaySet': str, 'essay_score1': np.int32, 'essay_score2': np.int32,
                            'EssayText': str})

    random_char = df.iloc[:1000]['EssayText']
    random_word = df.iloc[1000:2000]['EssayText']
    brown_char_ngram = df.iloc[2000:7000]['EssayText']
    brown_word_ngram = df.iloc[7000:12000]['EssayText']
    asap_char_ngram = df.iloc[12000:17000]['EssayText']
    asap_word_ngram = df.iloc[17000:22000]['EssayText']
    content_burst = df.iloc[22000:23000]['EssayText']
    shuffle = df.iloc[23000:24000]['EssayText']
    gpt_2 = df.iloc[24000:25001]['EssayText']
    adversarials = [random_char, random_word, brown_char_ngram, brown_word_ngram, asap_char_ngram, asap_word_ngram,
               content_burst, shuffle, gpt_2]
    for adv in adversarials:
        adv.dropna(inplace=True)
        test_sentences = adv.tolist()
        tokenized_text = [list(map(str.lower, nltk.tokenize.word_tokenize(sent))) for sent in test_sentences]
        test_data = [nltk.bigrams(t,  pad_right=True, pad_left=True, left_pad_symbol="<s>", right_pad_symbol="</s>") for t in tokenized_text]
        perplexity_list = []
        for i, test in enumerate(test_data):
            # print("PP({0}):{1}".format(test_sentences[i], model.perplexity(test)))
            perplexity_list.append(model.perplexity(test))
        print(perplexity_list.count(float("inf")) / len(perplexity_list)*100)