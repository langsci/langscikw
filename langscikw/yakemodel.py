"""Class for local YAKE! fork."""

from .yakefork import yake
# Functions that are shared between all models
from .kwe_toolkit import get_stopwords, read_text, save_keywords

class YakeExtractor():
    def __init__(self, corpus_path="", parameters=dict()):
        self.name = "yake"
        self.parameters = parameters
        self.corpus_path = corpus_path  # Only needed for cross-validation
        self.stopwords = get_stopwords()
    
    def extract_keywords(self, text_path, n, parameters=dict()):
        """Extracts keywords from a text and returns the top n results.

        Args:
            text_path (str): Path to the input file.
            n (int): Number of keywords to return.
            parameters (dict, optional): Keyword arguments dictionary for YAKE. 
            If empty, self.parameters is used.

        Raises:
            RuntimeError: If text is empty.

        Returns:
            list[tuple(str, float)]: Top n keywords and their score (lower is better).
        """
        
        if not parameters: parameters = self.parameters
        
        text = read_text(text_path)
        if not text: raise RuntimeError(f"Could not read input text from {text_path}. Wrong path or empty file?")
        
        # Object must be constructed here because it needs the n argument
        parameters["stopwords"] = self.stopwords
        kw_extractor = yake.KeywordExtractor(top=n, **parameters)
        keywords = kw_extractor.extract_keywords(text)
        return keywords

    def save_keywords(self, path, keywords):
        """Saves keywords to a newline-separated file."""
        
        save_keywords(path, keywords)


if __name__ == "__main__":
    # Adjust this 
    input_paths = ["data_tex_small/148.txt"]      # File/directory to extract keywords from
    keywords_path = "kw_yake.txt"                 # File to save keywords to
    n = 30                                        # Number of keywords
    model_parameters = {"ngram_range": (2, 2), "dedupLim": 0.75, "dedupFunc": "leve"}
    
    myyake = YakeExtractor()
    # yake.stopwords = {}       # Customize stop words list 
    for input in input_paths:
        kws = myyake.extract_keywords(input, n, model_parameters)
    
        if len(kws) < 41:
            for kw in kws:
                print(kw)
        else:
            myyake.save_keywords(keywords_path, kws)
