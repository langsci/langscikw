"""Class for sklearn's TF-IDF implementation."""

import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
# Functions that are shared between all models
from .kwe_toolkit import get_stopwords, read_text, save_keywords

class TfidfExtractor:
    def __init__(self, corpus_path="", parameters=dict()):
        """Loads a stop words list and validates the training corpus path. 

        Args:
            corpus_path (str, optional): Path to file(s) that should be used for training. 
            May be omitted and passed to build_corpus() instead.
            parameters (dict, optional): Model parameters. 
            May be omitted and passed to train() instead.
        """
        
        self.name = "tfidf"
        self.stop_words = get_stopwords()
        self.corpus = []
        self.model = None
        self.parameters = parameters
        self.corpus_path = corpus_path
        
    def build_corpus(self, path=""):
        """Constructs a corpus from self.corpus_path and saves to self.corpus. Includes preprocessing. 
        First tries to read all text files in path, then iterates over its subdirectories.
        
        Args:
            path (str, optional): Path training corpus (directory or single file). 
            If empty, self.corpus_path is used.

        Returns:
            list[str]: Corpus = list of documents. Each book is one string.
        """
        
        if not path: path = self.corpus_path
        corpus = []
        
        # Corpus from single file
        if os.path.isfile(path):
            corpus.append(read_text(path))
            return corpus
            
        # Corpus from directory
        if os.path.isdir(path):
            for _, dirs, files in os.walk(path):
                for entity in files + dirs:
                    corpus.append(read_text(os.path.join(path, entity)))
                    
                self.corpus = corpus
                return corpus
            
    def train(self, parameters=dict()):
        """Trains sklearn.TfidfVectorizer on self.corpus by calling fit_transform(). 
        Saves trained model to self.model.

        Args:
            parameters (dict, optional): Keyword argument dictionary (passed to TfidfVectorizer()). 
            If empty, self.parameters is used.
            
        Raises:
            RuntimeError: If self.corpus is empty.

        Returns:
            scipy.sparse.csr.csr_matrix: The document-term matrix.
        """
        
        if self.corpus == []:
            raise RuntimeError(f"Corpus is empty. Wrong path? {self.corpus_path}")
        if not parameters: parameters = self.parameters
        parameters["stop_words"] = list(self.stop_words)    # sklearn expects list, not set
        parameters["decode_error"] = "replace"              # Handle UnicodeDecodeError
        
        vectorizer = TfidfVectorizer(**parameters)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")                 # Ignore stop word list warning
            X = vectorizer.fit_transform(self.corpus)
        self.model = vectorizer
        return X
    
    def extract_keywords(self, text_path, n):
        """Extracts keywords from a text and returns the top n results.

        Args:
            text_path (str): Path to the input file.
            n (int): Number of keywords to return.

        Raises:
            RuntimeError: If text cannot be read.
            RuntimeError: If no trained model is available.

        Returns:
            list[tuple(str, float)]: Top n keywords and their score (higher is better).
        """
        
        text = read_text(text_path)
        if not text: raise RuntimeError(f"Could not read input text from {text_path}. Wrong path or empty file?")
        if not self.model: raise RuntimeError("No model. Please call train() before extract_keywords().")
        
        matrix = self.model.transform([text])
        feature_names = self.model.get_feature_names()
        keywords = self._extract_topn_from_matrix(matrix, feature_names, n)
        
        return keywords

    def _extract_topn_from_matrix(self, matrix, feature_names, n):
        """Helper function for extract_keywords. 
        Get the feature names and TF-IDF score of top n items from a feature matrix.
        Adapted from https://kavita-ganesan.com/extracting-keywords-from-text-tfidf/

        Args:
            matrix (scipy.sparse.csr.csr_matrix): Sparse matrix of (n_samples, n_features) (result of sklearn.transform()).
            feature_names (list[str]): Names of model features (= keywords).
            n (int): How many keywords to return.

        Returns:
            list[tuple]: List of (keyword, score) tuples.
        """
        
        # Sort matrix by score
        coo_matrix = matrix.tocoo()
        tuples = zip(coo_matrix.col, coo_matrix.data)
        sorted_vectors = sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
        
        results = []
        for i, score in sorted_vectors[:n]:
            results.append((feature_names[i], score))
        return results

    def save_keywords(self, keywords, path):
        """Saves keywords to a newline-separated file."""
        
        save_keywords(path, keywords)

    def save(self, thing, path):
        """Dumps thing to file using joblib. Used to pickle corpus or trained model."""
        
        joblib.dump(thing, path, compress=9)
        
    def load(self, path):
        """Loads something from file using joblib and returns it. Used for corpus or trained model."""
        
        return joblib.load(path)
        
    def reset(self):
        """Resets corpus and model to default values."""
        
        self.corpus = []
        self.model = None


if __name__ == "__main__":
    # Adjust this 
    input_paths = [""]                          # File/directory to extract keywords from
    keywords_path = "kw_tfidf.txt"              # File to save keywords to
    n = 30                                      # Number of keywords
    model_parameters = {"lowercase": True, "max_features": None, 
                        "ngram_range": (2, 2), "min_df": 0.05, "max_df": 0.8}
    
    tfidf = TfidfExtractor()
    # Don't use dir that contains input file! This is just an example
    tfidf.build_corpus("data_tex_small")
    
    # Save or load a corpus:
    # tfidf.save(tfidf.corpus, "corpora/corpus_tex.gz")
    # tfidf.corpus = tfidf.load("corpora/corpus_tex.gz")
    
    # tfidf.stop_words = {}                     # Customize stop words list
    tfidf.train(model_parameters)
    for file in input_paths:
        kws = tfidf.extract_keywords(file, n)
    
        if len(kws) < 41:
            for kw in kws:
                print(kw)
        else:
            tfidf.save_keywords(kws, keywords_path)
