import pandas as pd

import pandas as pd
import numpy as np

# aws
from utils_s3 import get_etf_holdings, list_keys

# gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim import corpora
from gensim import models
from gensim.utils import simple_preprocess
from gensim import similarities
from gensim.parsing.porter import PorterStemmer

labeled_data = pd.read_csv('model_inputs/FinancialPhraseBank-v1.0/Sentences_AllAgree.txt', sep='@',
                           engine='python', header=None, names=['lines', 'label'])

# tokenize and remove punctuation
labeled_data['lines'] = labeled_data['lines'].apply(lambda x: [w for w in simple_preprocess(x, deacc=True)])

# remove stopwords and uppercase words
labeled_data['lines'] = labeled_data['lines'].apply(lambda x: [remove_stopwords(w) for w in x])
labeled_data['lines'] = labeled_data['lines'].apply(lambda x: [w for w in x if (2 <= len(w) < 15)])

test = models.Word2Vec()
test.build_vocab(labeled_data['lines'], progress_per=1000)
test.train(labeled_data['lines'], total_examples=test.corpus_count, epochs=test.epochs)
test.init_sims(replace=True)
test.wv.most_similar(positive=["technology"])