"""Command line interface for langscikw."""

from collections import defaultdict
import os
import sys
from .langscikw import KWE

#           0     1    2      3         4         5          6
# python3 cli.py file [n] [corpus1] [corpus2] [keywords] --silent

install_path = os.path.dirname(os.path.abspath(__file__))

def main():
    usage = "USAGE: langscikw inputfile [n] [corpus1] [corpus2] [keywordslist] [--silent]"
    params = defaultdict(str)
    args = sys.argv

    try:
        params["input_file"] = args[1]
    except IndexError:
        print(usage)
        sys.exit(1)
    if not os.path.exists(args[1]): 
        print(f"Input file not found: {args[1]}")
        sys.exit(1)
        
    verbose = True
    if "--silent" in args: 
        verbose = False
        args = args[:-1]
        
    try:
        params["n"] = int(args[2])
        params["corpus1"] = args[3]
        params["corpus2"] = args[4]
        params["kwslist"] = args[5]
    except IndexError or ValueError:
        if not params["n"]: params["n"] = 300
        if not params["corpus1"]: params["corpus1"] = "corpus_tex.gz"
        if not params["corpus2"]: params["corpus2"] = "corpus_detexed.gz"
        if not params["kwslist"]: params["kwslist"] = os.path.join(install_path, "corpora", "keywordslist.txt")
    
    # Validate paths
    path_check = (os.path.exists(params["corpus1"]), os.path.exists(params["corpus2"]), os.path.isfile(params["kwslist"]))
    paths = (params["corpus1"], params["corpus2"], params["kwslist"])
    if not all(path_check):
        problem_idx = path_check.index(False)
        print(f"Corpus or keywords list doesn't exist: {paths[problem_idx]}")

        if problem_idx < 2:
            corpus_link = "https://github.com/langsci/langscikw-corpus/releases/tag/v1.0.1"
            print(f"If you want to use the langsci book corpora, download both from {corpus_link} and put them in current directory or provide path")
        else:
            repo_link = "https://github.com/langsci/langscikw"
            print(f"If you want to use the langsci keywords list, download it from {repo_link} and put it in current directory or provide path")
        sys.exit(1)
    
    if verbose: kwe = KWE()
    else: kwe = KWE(verbose=False)
    
    kwe.train(params["corpus1"], params["corpus2"])
    keywords = kwe.extract_keywords(args[1], n=params["n"], kw_corpus=params["kwslist"])
    
    for k in keywords:
        print(k)
    

if __name__ == "__main__":
    main()
