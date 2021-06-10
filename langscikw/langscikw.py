"""Extract bigram keywords from a given document."""

from time import time
import warnings

# Import models
from .yakemodel import YakeExtractor
from .tfidfmodel import TfidfExtractor
# # Import functions that are shared between all approaches
from .kwe_toolkit import distance, preprocess, save_keywords

JOBLIB_EXTS = ("z", "gz", "bz2", "xz", "lzma")   # https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html#joblib.dump

class KWE():
    def __init__(self, verbose=True):
        """Initializes the step 1 model (YAKE).

        Args:
            verbose (bool, optional): Whether to print status updates to console. Defaults to True.
        """
        
        self.name = "kwe"
        self.verbose = verbose
        self.model_step1 = YakeExtractor(parameters={"dedupLim": 0.75, "dedupFunc": "leve", "ngram_range": (2, 2)})
        self.model_step2 = None
        self.model_step3 = None
        self.max_steps = 1          # How many trained models are available
        self.parameters = dict()    # Only needed for evaluation, ignore
        
    def train(self, corpus_step2, corpus_step3=""):
        """Trains TF-IDF models for steps 2 and 3 and saves them in self.model_step2/3.

        Args:
            corpus_step2 (str): Path to corpus for step 2, i.e. raw TeX code.
            corpus_step3 (str, optional): Path to corpus for step 3, i.e. detexed documents, 
            if this model should be trained. Defaults to "" (= not trained).
        """
        
        step2_params = {"lowercase": False, "min_df": 0.05, "max_df": 0.8, "ngram_range": (2, 2)}
        step3_params = {"lowercase": True, "min_df": 0.02, "max_df": 0.8, "ngram_range": (2, 2)}
        
        self.model_step2 = self._train_model(corpus_step2, step2_params)
        self.model_step3 = self._train_model(corpus_step3, step3_params)
        
    def _train_model(self, corpus_path, params):
        """Helper function for train() that does the actual training.

        Args:
            corpus_path (str): Path to corpus file or directory.
            params (dict): Keyword argument dictionary.

        Returns:
            None: If corpus_path is empty.
            TfidfExtractor: Trained TF-IDF model.
        """
        
        if corpus_path == "": return
        steps = self.max_steps + 1
        tfidf = TfidfExtractor(parameters=params)
        
        # Check if corpus should be loaded from file or built fresh
        # Compressed files are loaded by joblib
        if any(corpus_path.split(".")[-1] == ext for ext in JOBLIB_EXTS):
            tfidf.corpus = tfidf.load(corpus_path)     # tfidf.load(), not self.load()!
        else:
            tfidf.build_corpus(corpus_path)
        if self.verbose: 
            print(f"Built corpus for step {steps}: {len(tfidf.corpus)} documents, {sum(len(doc) for doc in tfidf.corpus)} characters")
        
        tfidf.train()
        if self.verbose: print(f"Trained model for step {steps}")
        self.max_steps = steps
        return tfidf
    
    def extract_keywords(self, path, n=300, kw_corpus="corpora/keywordslist.txt", dedup_lim=0.85):
        """Extracts keywords from a document in three steps:
        1. Book-specific keywords, using YAKE algorithm
        2. Broad topic keywords, using TF-IDF algorithm
        3. Filler keywords that already appeared in other books, using TF-IDF algorithm.

        Args:
            path (str): Path to input text file.
            n (int, optional): Number of keywords to extract. Defaults to 200.
            kw_corpus (str, optional): Path to keywords corpus for step 3. Defaults to "corpora/keywordslist.txt".
            dedup_lim (float, optional): Deduplication threshold (Jaro-Winkler) [0,1]. Defaults to 0.85.

        Returns:
            list[str]: Alphabetically sorted list of keywords.
        """
        
        if self.verbose: print(f"Extracting {n} keywords from {path} in {self.max_steps} steps")
        
        # Step 1: YAKE
        keywords = self.model_step1.extract_keywords(path, n)
        if self.verbose: print("Step 1 done")
        
        # Step 2: Fill with TF-IDF on raw TeX documents
        if self.max_steps > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")     # Ignore stop words list warning
                keywords += self.model_step2.extract_keywords(path, n)
            if self.verbose: print("Step 2 done")
            
        # Step 3: Fill with TF-IDF on detexed documents, only previously seen keywords
        if self.max_steps > 2:
            keywords += self._extract_filler_kws(path, n, kw_corpus)
            if self.verbose: print("Step 3 done")
            
        assert [type(k) == tuple for k in keywords] or keywords == []
        keywords = self._deduplicate(keywords, dedup_lim)
        if self.verbose: print("Postprocessing done\n")
        return sorted(keywords)
    
    def _extract_filler_kws(self, text_path, n, kw_corpus):
        """Helper function for extraction step 3. Extracts keywords and compares 
        them against a corpus of previously seen keywords.

        Args:
            text_path (str): Path to input file.
            n (int): Number of keywords to extract (will extract 1.5*n since only 
            a small portion will be kept).
            kw_corpus (str): Path to keyword corpus file.

        Returns:
            list[tuple(str,float)]: Extracted keywords that also appeared in the corpus and their scores.
        """
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # Ignore stop words list warning
            kws = self.model_step3.extract_keywords(text_path, int(n*1.5))

        # Read and preprocess corpus
        with open(kw_corpus, errors="replace") as f:
            corpus = f.read().split("\n")[:-1]  # [:-1] removes empty string
        corpus = preprocess(" § ".join(corpus)).split(" § ") # arbitrary delimiter
        corpus = set(corpus)
        
        # Return kws that appear in the corpus
        kws_union = [k for k in kws if k[0] in corpus]
        kws_union += [(k[0].lower(), k[1]) for k in kws if k[0].lower() in corpus]
        return kws_union
    
    def _deduplicate(self, kws, threshold=0.85):
        """Deduplicates/postprocesses the list of keywords.

        Args:
            kws (list[tuple(str,float)]): Keywords list.
            threshold (float, optional): Deduplication threshold (Jaro-Winkler) [0,1]. Defaults to 0.85.

        Returns:
            list[str]: Deduplicated list (without scores).
        """
        
        final_kws = []
        all_kws = {k[0] for k in kws}   # Basic deduplication
        
        # Include only those kws that are not similar (edit distance) to others
        for candidate in all_kws:
            tokens = candidate.split()
            # Ignore all candidates with non-alphabetic characters (e.g. numbers)
            # "-" is allowed though
            if not all(t.isalpha() for t in tokens) and not any("-" in t for t in tokens): continue
            # Ignore all candidates that consist of only 1 token (e.g. "person person")
            if tokens.count(tokens[0]) > 2: continue
            # Ignore ngrams of form "x X" (second token capitalized)
            try: 
                if tokens[0].islower() and tokens[1].isupper(): continue
            except IndexError: pass
            
            include = True
            for kw in final_kws:
                if distance(candidate, kw) > threshold: 
                    include = False
                    break
            if include: final_kws.append(candidate)
        return final_kws
    
    def save_keywords(self, path, keywords):
        """Saves keywords to a newline-separated file."""
        
        save_keywords(path, keywords)
        
    def reset(self):
        """Resets instance variables to default values. This will delete all trained models."""
        
        self.model_step1 = YakeExtractor(parameters={"dedupLim": 0.75, "dedupFunc": "leve", "ngram_range": (2, 2)})
        self.model_step2 = None
        self.model_step3 = None
        self.max_steps = 1
        self.parameter = dict()


if __name__ == "__main__":
    input_paths = ["148.txt"]           # File/directory to extract keywords from
    keywords_path = "keywords.txt"      # File to save keywords to
    n = 300                             # Number of keywords
    
    t0 = time()
    kwe = KWE()
    kwe.train("corpora/corpus_tex.gz", "corpora/corpus_detexed.gz")
    
    for input in input_paths:
        kws = kwe.extract_keywords(input, n=250)
        if len(kws) < 41:
            for kw in kws:
                print(kw)
        else:
            kwe.save_keywords(keywords_path, kws)
            
    print(f"Ran in {round(time()-t0)} seconds")
