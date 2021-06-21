import json

import numpy as np
import pandas as pd

from scipy.spatial.distance import cosine

from natasha import Doc, NewsEmbedding, NewsMorphTagger, Segmenter, MorphVocab
from navec import Navec


def _listify(x):
    if not isinstance(x, (list, tuple)):
        x = list(x)
    return x


class BeerEmbedding:
    def __init__(self):
        with open('./beer_features.json', 'r') as f:
            self.feature_space = json.load(f)
        self._word_embedding = Navec.load('navec_hudlit_v1_12B_500K_300d_100q.tar')
        self._segmenter = Segmenter()
        self._morph_tagger = NewsMorphTagger(NewsEmbedding())
        self._morph_vocab = MorphVocab()

        self._build_beer_table()
        self._precompute_embeddings()

    def _build_beer_table(self):
        with open('./beer_data.json', 'r') as f:
            data = pd.read_json(f)
        names = []
        features = []
        for key in self.feature_space.keys():
            data[key] = data[key].apply(
                lambda x: list(map(str.lower, x)) if x is not np.nan and x is not None else [])
        for _, row in data.iterrows():
            names.append(row['name'])
            features.append(self.featurize(row))
        features = np.vstack(features)
        self._names = names
        self._features = features

    def _precompute_embeddings(self):
        emb = []
        for values in self.feature_space.values():
            for v in values:
                tokens = self._preprocess_sentence(v)
                val_emb = []
                for token in tokens:
                    val_emb.append(self._word_embedding[token])
                emb.append(val_emb)
        self._feature_embeddings = emb

    def featurize(self, row):
        emb = []
        for k, v in self.feature_space.items():
            arr = np.zeros_like(v, dtype=float)
            values = _listify(row[k])
            for v_ in values:
                if v_ in v:
                    arr[v.index(v_)] = 1
            emb.append(arr)
        emb = np.hstack(emb)
        return emb

    def _preprocess_sentence(self, sentence):
        doc = Doc(sentence)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)
        tokens = [_ for _ in doc.tokens if _.pos in {'NOUN', 'ADJ', 'PROPN'}]
        for token in tokens:
            token.lemmatize(self._morph_vocab)
        tokens = [_.lemma for _ in tokens]
        return tokens

    def _cosine_similarity(self, x, y):
        return 1 - cosine(x, y)

    def match(self, sentence):
        tokens = self._preprocess_sentence(sentence)


