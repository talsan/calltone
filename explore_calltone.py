import pandas as pd

pd.set_option('max_columns', None)
pd.set_option("max_rows", 100)

import csv

# aws
from utils_s3 import get_etf_holdings, list_keys

# gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora
from gensim import models
from gensim.utils import tokenize
from gensim.parsing.porter import PorterStemmer
from gensim.summarization.textcleaner import split_sentences

# textblob
from textblob import TextBlob
import numpy as np
from scipy.stats.mstats import winsorize
from scipy.stats import zscore
import itertools
from pandas.tseries.offsets import MonthEnd
from datetime import datetime

# get list of quarterly calls (events) generated from download_transcripts.py
calls_raw = pd.read_csv('./extracts/foolcalls_extract_20201001.csv')

# only calls from 2019/2020
calls_recent = calls_raw.loc[calls_raw['fiscal_period_year'].isin([2019, 2020]), :]

# remove statement_types that are either unknown or operator (i.e. keeping P, Q, A)
calls_recent_PQA = calls_recent.loc[calls_raw['statement_type'].isin(['P', 'Q', 'A'])]

# only analyze 1000 largest stocks from the R3000
russell_500 = get_etf_holdings('IWV', '2020-07-31').sort_values('weight', ascending=False).head(500)
calls_recent_PQA_R500 = calls_recent_PQA.merge(russell_500, on='ticker', how='inner')

# remove rows with missing text (nan)
calls_recent_PQA_R500 = calls_recent_PQA_R500.dropna(subset=['text'])

# order it in a way that's easier to look at
calls_recent_PQA_R500 = calls_recent_PQA_R500.sort_values(['cid', 'statement_num']).reset_index(drop=True)

# todo coverage analysis

# write out to file for finBert sentiment analysis
'''
output = []
for i, row in calls_recent_PQA_R500.iterrows():
    this_output = pd.DataFrame({'sentence': split_sentences(row['text'])})
    this_output['cid'] = row['cid']
    this_output['statement_num'] = row['statement_num']
    this_output = this_output[['cid', 'statement_num', 'sentence']]
    output.append(this_output)
output_df = pd.concat(output)
output_df.to_csv('./output/r1000_sentences.csv', index=False)
output_df['sentence'].to_csv('./output/r1000_sentences_text_only.csv', sep='\n', index=False, header=False,
                             quoting=csv.QUOTE_NONE)
'''

# allow process to run
'''
conda activate finbert
(finbert) finbert_dir>
python predict.py 
--text_path R500_sentences_text_only.csv 
--output_dir output/ 
--model_path models/sentiment/sentiment.tar.gz''
'''

# read classified finBert Sentences from output
sent_classified = pd.read_csv('./output/r1000_sentences_classified.csv')
sent_map = pd.read_csv('./output/r1000_sentences.csv')
sent_finBert = sent_map[['cid', 'statement_num']].join(sent_classified)


def calc_call_sentiment(predictions):
    sent_count = len(predictions)
    pos = predictions[predictions == 'positive'].count()
    neg = predictions[predictions == 'negative'].count()
    return pd.Series({'finBert_pos': pos / sent_count,
                      'finBert_neg': neg / sent_count,
                      'finBert_sent': (pos - neg) / sent_count
                      })


# take mean from sentences into one score per call
calls_sent_finBert = sent_finBert.groupby(['cid']).apply(lambda x: calc_call_sentiment(x['prediction']))
calls_sent_finBert['finBert_sent_mean'] = sent_finBert.groupby(['cid'])['sentiment_score'].mean()

# join text by call (i.e. combine individual statements from the same call)
calls = calls_recent_PQA_R500.loc[:, ['cid', 'text']].groupby(['cid'])['text'].apply(
    lambda x: ''.join(x)).reset_index()

# -----------------------------------------------------------------------------------------------
# calc text blob sentiment
calls_textblob = pd.DataFrame({'cid': calls['cid'],
                               'sent_tb': calls['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
                               })
# -----------------------------------------------------------------------------------------------
# build dictionary sentiment
pos_list = pd.read_csv('./model_inputs/lm_word_lists/lm_positive.csv', header=None)[0].str.lower().to_list()
neg_list = pd.read_csv('./model_inputs/lm_word_lists/lm_negative.csv', header=None)[0].str.lower().to_list()
pos_neg = pos_list + neg_list

# tokenize and remove punctuation
calls['text'] = calls['text'].apply(lambda x: [w for w in tokenize(x, deacc=True)])

# remove stopwords and uppercase words
calls['text'] = calls['text'].apply(lambda x: [remove_stopwords(w) for w in x])
calls['text'] = calls['text'].apply(lambda x: [w for w in x
                                               if (2 < len(w) < 15) and (w.islower())])

# stemming (porter)
p = PorterStemmer()
calls['text'] = calls['text'].apply(lambda x: p.stem_documents(x))

# create dictionary object
dictionary = corpora.Dictionary(calls['text'])

agg_word_freq = {dictionary[k]: v for k, v in dictionary.cfs.items()}
agg_word_freq = pd.DataFrame({'token': agg_word_freq.keys(), 'cf': agg_word_freq.values()})

agg_doc_freq = {dictionary[k]: v for k, v in dictionary.dfs.items()}
agg_doc_freq = pd.DataFrame({'token': agg_doc_freq.keys(), 'df': agg_doc_freq.values()})

agg_freq = agg_word_freq.merge(agg_doc_freq, on='token')
agg_freq.sort_values(by='cf', ascending=False, inplace=True)

# filter extremes
# dictionary.filter_extremes(no_below=2, no_above=0.5)

# bag-of-words transformation
corpus = [dictionary.doc2bow(text) for text in calls['text']]

# tfidf transformation
tfidf = models.TfidfModel(corpus)  # fit model
corpus_tfidf = tfidf[corpus]  # apply model

# count words per call
word_tfidf = [[(dictionary[id], count) for id, count in line] for line in corpus_tfidf]
word_cf = [[(dictionary[id], count) for id, count in line] for line in corpus]

# calculate sentiment
pos_stemmed = list(set(p.stem_documents(pos_list)))
neg_stemmed = list(set(p.stem_documents(neg_list)))

# calculate tfidf sentiment
sent_tfidf = []
for call in word_tfidf:
    pos = sum([tfidf for tkn, tfidf in call if tkn in pos_stemmed])
    neg = sum([tfidf for tkn, tfidf in call if tkn in neg_stemmed])

    sent_tfidf.append({'pos_tfidf': pos / (pos + neg),
                       'neg_tfidf': neg / (pos + neg),
                       'sent_tfidf': (pos - neg) / (pos + neg)})

# calculate count frequency sentiment
sent_cf = []
for call in word_cf:
    word_count = sum([cf for tkn, cf in call])
    pos = sum([cf for tkn, cf in call if tkn in pos_stemmed]) / word_count
    neg = sum([cf for tkn, cf in call if tkn in neg_stemmed]) / word_count

    sent_cf.append({'pos_cf': pos,
                    'neg_cf': neg,
                    'sent_cf': pos - neg})

# combine both lm sentiment models
calls_sent_lm = pd.concat([calls, pd.DataFrame(sent_tfidf), pd.DataFrame(sent_cf)], axis=1)
calls_sent_lm = calls_sent_lm.sort_values(by='sent_cf', ascending=False)

# combine with finbert
calls_sent = calls_sent_lm.merge(calls_sent_finBert, how='inner', on='cid')

# combine with texblob
calls_sent = calls_sent.merge(calls_textblob, how='inner', on='cid')
# add back call info
call_info = calls_recent_PQA_R500.groupby('cid').head(1).reset_index(drop=True)
calls_sent_w_info = call_info.merge(calls_sent, how='inner', on='cid')
calls_sent_w_info = calls_sent_w_info.loc[
    (calls_sent_w_info['call_date'] >= '2019-01-01') & (calls_sent_w_info['call_date'] <= '2020-08-31'),
    ['cid', 'ticker', 'call_date', 'sector', 'duration_minutes', 'sent_cf', 'sent_tb', 'finBert_sent_mean']]
calls_sent_w_info['yyyymm'] = calls_sent_w_info['call_date'].str[0:7]

# -----------------------------------------------------------------------------------------------
# build signal
def expandgrid(*itrs):
    product = list(itertools.product(*itrs))
    return {'Var{}'.format(i + 1): [x[i] for x in product] for i in range(len(itrs))}


# monthly dates
monthly_dates = pd.date_range(min(call_info['call_date']), max(call_info['call_date']), freq='M')
signal_scores = pd.DataFrame(expandgrid(list(set(call_info['ticker'])), list(monthly_dates))).rename(
    columns={'Var1': 'ticker', 'Var2': 'eomdate'})
signal_scores['eomdate'] = signal_scores['eomdate'].dt.strftime('%Y-%m-%d')

# call_info['eomdate'] = (pd.to_datetime(call_info['call_date']) + MonthEnd(1)).dt.strftime('%Y-%m-%d')

# build signals
call_info_monthly = []
for i, row in signal_scores.iterrows():
    # get latest call for a given month that is still within 180 days
    latest_call = (call_info.loc[(call_info['call_date'] < row['eomdate'])
                                 & (call_info['ticker'] == row['ticker']), :]
                   .sort_values(by='call_date', ascending=False)
                   .head(1))
    if not latest_call.empty:
        if (datetime.strptime(row['eomdate'], '%Y-%m-%d') - datetime.strptime(latest_call['call_date'].iloc[0],
                                                                              '%Y-%m-%d')).days <= 125:
            latest_call['eomdate'] = row['eomdate']
            call_info_monthly.append(latest_call)

call_info_monthly = pd.concat(call_info_monthly,ignore_index=True)

call_sent_monthly = (call_info_monthly[['ticker','eomdate','cid','call_date', 'sector','duration_minutes']]
    .merge(calls_sent[['cid', 'sent_tfidf', 'pos_cf',
       'neg_cf', 'sent_cf', 'finBert_pos', 'finBert_neg', 'finBert_sent',
       'finBert_sent_mean', 'sent_tb']], how='inner',on='cid')).reset_index(drop=True)

call_sent_monthly_long = call_sent_monthly.melt(id_vars=['ticker','eomdate','cid','call_date', 'sector'])

call_sent_by_month2 = (call_sent_monthly_long.loc[call_sent_monthly_long['variable']=='duration_minutes',:]
                       .groupby(['eomdate', 'variable'])['value']
                       .median().unstack().plot(subplots=True))

call_sent_by_month2 = call_sent_monthly_long.groupby(['eomdate', 'variable']).count()


# create ticker-monthend skeleton
# loop through each ticker/month end and find the most recent earnings call within 180 days
# populate the 3 sentiment scores to that ticker/month-end

# calc momentum
# go through teh signal file and calculate the returns using hte same calc as process as before


# -----------------------------------------------------------------------------------------------


# ceo language cto
# longer calls during covid

# value_vs_growth
russell_1000G = get_etf_holdings('IWF', '2020-07-31').sort_values('weight', ascending=False)
russell_1000V = get_etf_holdings('IWD', '2020-07-31').sort_values('weight', ascending=False)
calls_sent_w_info['style'] = 'V'
calls_sent_w_info.loc[calls_sent_w_info['ticker'].isin(russell_1000G['ticker']), 'style'] = 'G'
calls_sent_w_info.loc[calls_sent_w_info['ticker'].isin(russell_1000V['ticker']), 'style'] = 'V'

# words in value vs growth?

calls_sent_w_info_long = calls_sent_w_info.melt(id_vars=['cid', 'ticker', 'call_date', 'sector', 'yyyymm', 'style'])
call_sent_by_month = calls_sent_w_info_long.groupby(['yyyymm', 'variable'])['value'].mean()


def z_score(values):
    return (values - values.mean()) / values.std()


calls_sent_w_info_long['zscore'] = calls_sent_w_info_long.groupby(['variable'])['value'].transform(lambda x: z_score(x))

call_sent_by_month = calls_sent_w_info_long.groupby(['yyyymm', 'variable'])['zscore'].mean().unstack().plot()
call_sent_by_month = calls_sent_w_info_long.groupby(['yyyymm', 'variable'])['zscore'].mean().unstack().plot()
call_sent_by_style = calls_sent_w_info_long.groupby(['style', 'variable'])['zscore'].mean().unstack().plot(kind='bar')
call_sent_by_sector = calls_sent_w_info_long.groupby(['sector', 'variable'])['zscore'].mean().unstack().plot(kind='bar')
calls_sent_w_info_long.groupby(['style']).count()
calls_sent_w_info_long.groupby(['yyyymm']).count()
calls_sent_w_info_long.groupby(['sector']).count()

# get returns for all the names
# ------------------------------------------------------------------------------------
all_rets_td = pd.read_csv('./extracts/returns_extract_rundate_20200930.csv')
all_rets_td.loc[all_rets_td['return'] > 1, 'return'] = 1

daily_dates = pd.DataFrame({'asofdate':pd.date_range(min(all_rets_td['asofdate']), max(all_rets_td['asofdate']), freq='D')})
daily_dates['asofdate'] = daily_dates['asofdate'].dt.strftime('%Y-%m-%d')

# daily dates
daily_dates = pd.date_range(min(call_info['call_date']), max(call_info['call_date']), freq='D')
all_rets_d = pd.DataFrame(expandgrid(list(set(all_rets_td['ticker'])), list(daily_dates))).rename(
    columns={'Var1': 'ticker', 'Var2': 'asofdate'})
all_rets_d['asofdate'] = all_rets_d['asofdate'].dt.strftime('%Y-%m-%d')

all_rets = all_rets_d.merge(all_rets_td,on=['ticker','asofdate'],how='left')
all_rets['return'] = all_rets['return'].fillna(0)
all_rets.sort_values(['ticker', 'asofdate'],inplace=True)
all_rets['cml_ret'] = all_rets.groupby('ticker')['return'].transform(lambda x: (1 + x).cumprod())
all_rets.reset_index(drop=True, inplace=True)

# output dataframe
# calculate lag/fwd returns for every stock at every month end


# ------------------------------------------------------------------------------------


all_rets_univ = all_rets.sort_values(by='asofdate').groupby('asofdate')['return'].mean().reset_index()
all_rets_univ['cml_ret'] = all_rets_univ['return'].transform(lambda x: (1 + x).cumprod())
all_rets_univ.index = all_rets_univ['asofdate']

ret_horizons_list = [7, 31, 92, 182, 365]
ret_cols = [f't-{td}' for td in ret_horizons_list] + [f't+{td}' for td in ret_horizons_list]

def calc_univ_ret(all_rets_univ, start_date, end_date):
    return all_rets_univ['cml_ret'][end_date] / all_rets_univ['cml_ret'][start_date] - 1


def calc_univ_ret_sum(all_rets_univ, start_date, end_date):
    return all_rets_univ.loc[
        (all_rets_univ['asofdate'] >= start_date) & (all_rets_univ['asofdate'] <= end_date), 'return'].sum()


def calc_active_returns(ticker_rets, univ_rets, call_date_i, ret_horizons_list):
    ret_dict = {'geo': {'type': 'geo'}, 'arith': {'type': 'arith'}}
    # back returns
    for td in ret_horizons_list:
        ret_dict['geo'].update({f't-{td}': np.NaN})
        ret_dict['arith'].update({f't-{td}': np.NaN})
        try:
            univ_ret = calc_univ_ret(univ_rets,
                                     ticker_rets['asofdate'][call_date_i - td],
                                     ticker_rets['asofdate'][call_date_i])
            asset_ret = ticker_rets['cml_ret'][call_date_i] / ticker_rets['cml_ret'][call_date_i - td] - 1
            ret_dict['geo'][f't-{td}'] = asset_ret - univ_ret

            univ_ret_sum = calc_univ_ret_sum(univ_rets,
                                             ticker_rets['asofdate'][call_date_i - td],
                                             ticker_rets['asofdate'][call_date_i])
            asset_ret_sum = ticker_rets['return'][(call_date_i - td):call_date_i].sum()
            ret_dict['arith'][f't-{td}'] = asset_ret_sum - univ_ret_sum
        except KeyError:
            pass
    # fwd returns
    for td in ret_horizons_list:
        ret_dict['geo'].update({f't+{td}': np.NaN})
        ret_dict['arith'].update({f't+{td}': np.NaN})
        try:
            univ_ret = calc_univ_ret(univ_rets,
                                     ticker_rets['asofdate'][call_date_i],
                                     ticker_rets['asofdate'][call_date_i + td])
            asset_ret = ticker_rets['cml_ret'][call_date_i + td] / ticker_rets['cml_ret'][call_date_i] - 1
            ret_dict['geo'][f't+{td}'] = asset_ret - univ_ret

            univ_ret_sum = calc_univ_ret_sum(univ_rets,
                                             ticker_rets['asofdate'][call_date_i],
                                             ticker_rets['asofdate'][call_date_i + td])

            asset_ret_sum = ticker_rets['return'][call_date_i:(call_date_i + td)].sum()
            ret_dict['arith'][f't+{td}'] = asset_ret_sum - univ_ret_sum
        except KeyError:
            pass

    return pd.concat([pd.DataFrame(ret_dict['geo'], index=[0]),
                      pd.DataFrame(ret_dict['arith'], index=[0])])



# compute returns, using the monthly dates
ret_horizons_list = [7, 31, 92, 182, 365]
output_monthly = []
for i, row in call_info_monthly.iterrows():
    print(i / call_info_monthly.shape[0])
    ticker_rets = all_rets.loc[all_rets['ticker'] == row['ticker'], :].reset_index(drop=True)
    if ticker_rets.shape[0] > 0:
        call_ret_idx = ticker_rets[ticker_rets['asofdate'] == row['eomdate']]
        if call_ret_idx.shape[0] > 0:
            call_date_i = call_ret_idx.index[0]
            output_monthly.append(calc_active_returns(ticker_rets, all_rets_univ, call_date_i, ret_horizons_list))

output_monthly_arith = [o.loc[o['type'] == 'arith', :] for o in output_monthly]
call_rets_monthly = pd.concat(output_monthly_arith, ignore_index=True)
call_rets_monthly = pd.concat([call_info, call_rets_monthly], axis=1)

calls_sent_w_rets_monthly = call_rets_monthly.merge(call_sent_monthly, how='inner', on='cid')
sent_ret_corr_monthly = calls_sent_w_rets_monthly.loc[calls_sent_w_rets_monthly['eomdate'] <= '2020-01-31',
                                       ['sent_cf', 'sent_tb', 'finBert_sent_mean'] + ret_cols].corr(method='spearman')
sent_ret_corrs = calls_sent_w_rets_monthly[['duration_minutes_x', 'sent_cf', 'sent_tb', 'finBert_sent_mean'] + ret_cols].corr(
    method='spearman')

calls_sent_w_rets_monthly = call_rets_monthly.merge(call_sent_monthly_long, how='inner', on='cid')

# compute returns, anchored by the call date (more event-study like)
ret_horizons_list = [5, 21, 63, 127, 255]
output = []
for i, row in call_info.iterrows():
    print(i / call_info.shape[0])
    ticker_rets = all_rets.loc[all_rets['ticker'] == row['ticker'], :].reset_index(drop=True)
    if ticker_rets.shape[0] > 0:
        call_ret_idx = ticker_rets[ticker_rets['asofdate'] == row['call_date']]
        if call_ret_idx.shape[0] > 0:
            call_date_i = call_ret_idx.index[0]
            output.append(calc_active_returns(ticker_rets, all_rets_univ, call_date_i, ret_horizons_list))

output_arith = [o.loc[o['type'] == 'arith', :] for o in output]
# agg returns before and after call
call_rets = pd.concat(output_arith, ignore_index=True)
call_rets = pd.concat([call_info, call_rets], axis=1)

calls_sent_w_rets = call_rets.merge(calls_sent, how='inner', on='cid')

sent_corrs = calls_sent[['pos_cf', 'neg_cf', 'sent_cf', 'pos_tfidf', 'neg_tfidf', 'sent_tfidf',
                         'sent_tb', 'finBert_pos', 'finBert_neg', 'finBert_sent', 'finBert_sent_mean']].corr()
sent_ret_corrs = calls_sent_w_rets.loc[calls_sent_w_rets['call_date'] <= '2020-01-31',
                                       ['sent_cf', 'sent_tb', 'finBert_sent_mean'] + ret_cols].corr()
sent_ret_corrs = calls_sent_w_rets[['duration_minutes', 'sent_cf', 'sent_tb', 'finBert_sent_mean'] + ret_cols].corr(
    method='spearman')
calls_sent = calls_sent.sort_values(by='prediction', ascending=False)

calls_sent_w_rets['sent_cf_q'] = pd.qcut(calls_sent_w_rets['sent_cf'], 5, labels=False)
calls_sent_w_rets['sent_tb_q'] = pd.qcut(calls_sent_w_rets['sent_tb'], 5, labels=False)
calls_sent_w_rets['finBert_sent_mean_q'] = pd.qcut(calls_sent_w_rets['finBert_sent_mean'], 5, labels=False)

test = calls_sent_w_rets.loc[calls_sent_w_rets['call_date'] <= '2020-01-31',
                             ['cid', 'sent_cf_q', 'sent_tb_q', 'finBert_sent_mean_q'] + ret_cols].melt(
    id_vars=['cid'] + ret_cols)
test.groupby(['variable', 'value'])[ret_cols].sum()

test = calls_sent_w_rets[['cid', 'sent_cf_q', 'sent_tb_q', 'finBert_sent_mean_q'] + ret_cols].melt(
    id_vars=['cid'] + ret_cols)
test.groupby(['variable', 'value'])[ret_cols].sum()

# time series

test = models.Word2Vec()
test.build_vocab(calls['text'], progress_per=1000)
test.train(calls['text'], total_examples=test.corpus_count, epochs=test.epochs)
test.init_sims(replace=True)
test.wv.most_similar(positive=["fear"])
