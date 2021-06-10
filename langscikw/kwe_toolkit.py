"""Collection of functions that are shared by multiple scripts/algorithms"""

from jellyfish import jaro_winkler
import os
import re

def compare(keywords, gold_set, fuzzy=True):
    """Compares two sets of keywords and retrieves exact or "fuzzy" matches. 
    Fuzzy matches are exact matches + those within a certain distance measure value.
    Sets are used instead of lists since the order is not important (kw lists are sorted alphabetically).

    Args:
        keywords (list | set | list(tuple(float, str))): Keywords, e.g. from a model.
        gold_set (set): Set of kws to compare against, e.g. a gold standard.
        fuzzy (bool, optional): Whether to retrieve fuzzy and exact (True) or only exact (False) 
        matches. Defaults to True.

    Returns:
        set: Exact (identical) or fuzzy matches.
    """
    
    if type(keywords) == set:
        kw_set = keywords
    else:
        try:
            kw_set = {k[0] for k in keywords}
        except TypeError:
            kw_set = set(keywords)
        else:
            raise TypeError(f"Keywords must be list or set, not {type(keywords)}.")
    
    ## Precision & recall
    # precision = len(gold_set & kw_set) / len(kw_set)
    # recall = len(gold_set & kw_set) / len(gold_set)
    # print(f"Precision: {precision:.3f}, recall: {recall:.3f}")
    
    # Exact matches
    if not fuzzy:
        exact_matches = gold_set & kw_set
        return sorted(list(exact_matches))
    
    # Fuzzy matches, i.e. if the extracted kw is contained in the gold kw
    # But they are not identical
    fuzzy_matches = set()
    for k in kw_set:
        for g in gold_set:
            # Old method:
            # if k == g or k in g or g in k:
            #     fuzzy_matches.add((k,g))
            #     break   # Each k may only be added once
            # New method:
            if distance(k, g) >= 0.9:
                fuzzy_matches.add((k,g))
                break   # Each k may only be added once
    return sorted(list(fuzzy_matches))

def distance(a, b):
    """Returns the distance between two strings. Uses jellyfish.jaro_winkler()."""
    
    return jaro_winkler(a, b)   # Using jellyfish

def get_stopwords():
    """Reads stopwords from path and returns as set."""
    
    # Read into set for deduplication
    stopwords = set()
    install_path = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(install_path, "corpora", "stopwords.txt")) as f: 
        for line in f:
            stopwords.add(line.strip())
    return stopwords

def preprocess(text):
    """Preprocesses a string and returns it. 
    Currently no preprocessing is performed since it did not increase performance, 
    but naive lemmatization using regex is available.
    """
    
    # Naive lemmatization
    # text = re.sub(r"([a-z]{3,4}[a-z]+)ies(\.|,|:| )", r"\1y\2", text)
    # text = re.sub(r"([a-z]{3,4}[a-z]+)s(\.|,|:| )", r"\1\2", text)
    return text

def read_text(path):
    """Reads text from a single file or a directory and returns all text as one string."""
    
    # Directory
    if os.path.isdir(path):
        fulltext = ""
        files = _find_text_files(os.listdir(path))
        files = _sort_chapters(files)    # Move intro chapter to the front
        if files == []:
            raise print(f"Directory '{path}' does not contain any text files.")
        for file in files:
            try:
                with open(os.path.join(path, file), errors="replace") as f:
                    filetext = f.read()
            except Exception as excp:
                print(f"Exception for file {path}/{file}: ", excp)
                filetext = ""
            filetext = preprocess(filetext)
            fulltext += filetext + "\n\n"
        if not fulltext: print(f"No text found in {path}. Is directory empty?")
        return fulltext
    
    # Single file
    elif os.path.isfile(path):
        path = _find_text_files(path)
        try:
            with open(path, errors="replace") as f:
                text = f.read()
        except Exception as excp:
            print(f"Exception for file {path}: ", excp)
            text = ""
        text = preprocess(text)
        if not text: print(f"No text found in {path}. Is file empty?")
        return text
    
    # Do nothing if path is not legal since _find_text_files() prints a warning
    elif _find_text_files(path) == "" or _find_text_files(path) == []:
        return ""

    else:
        print(f"Cannot read text from {path}")
        return ""
    
def _find_text_files(files, legal_files=(".txt", ".tex")):
    """Finds text files and returns them.

    Args:
        files (list(str) | str): Filename or list of filesnames.
        legal_files (tuple, optional): Filenames to look for. Defaults to (".txt", ".tex").

    Raises:
        FileNotFoundError: If input is string and does not have a legal file extension.
        TypeError: If files is not of type list or str.
        
    Returns:
        list(str) | str: Legal files.
    """
    
    # Directory
    if type(files) == list:
        files = [f for f in files if f[-4:] in legal_files]
        if files: 
            return files
        else: 
            print(f"Cannot find legal files ({legal_files}) in file list. Ignoring.")
            return []
        
    # Single file
    elif type(files) == str:
        if files[-4:] in legal_files: 
            return files
        else: 
            raise FileNotFoundError(f"File {files} is not legal. Only {legal_files} files allowed.")
        
    else:
        raise TypeError(f"Files must be list or string, not {type(files)}.")
    
def _sort_chapters(files):
    """Sort tex files as well as possible,
    i.e. alphabetically, but starting with the introduction.
    Important since YAKE! weighs terms at the beginning of the file."""
    
    files = sorted(files)
    c = 0
    for f in files:
        if "intro" in f or "einleitung" in f or "1" in f:
            files.remove(f)
            files.insert(c, f)
            c += 1
        elif "conclusion" in f:
            files.remove(f)
            files.append(f)
    return files
    
def save_keywords(path, keywords):
    """Saves keywords to a newline-separated file"""
    
    if type(keywords[0]) == tuple:
        with open(path, mode="w+") as f:
            for k in keywords:
                f.write(k[0] + "\t" + str(k[1]) + "\n")
    elif type(keywords[0]) == str:
        with open(path, mode="w+") as f:
            for k in keywords:
                f.write(k + "\n")
    else:
        raise TypeError("Expected list of tuples or strings")