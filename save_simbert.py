from collections import Counter
import bert4keras
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
#from bert4keras.snippets import uniout, open
from keras.models import Model
from bert4keras.tokenizers import Tokenizer, load_vocab
import tensorflow as tf
import pickle

with open("frozen_graph.pkl","rb") as f:
    tf.saved_model.save(pickle.load(f),"./simbert_model/1")
