# Keyword extraction from langsci publications

```langscikw``` is a Python package and command line tool for bigram keyterm extraction. It is optimized for long, English, linguistic publications and can also be applied to TeX code.

Keyword extraction is done in three steps. No preprocessing is needed.
* **Step 1:** KWE from the input document using [YAKE](https://pypi.org/project/yake/). This is the simplest step as it doesn't need a corpus and should extract the most important keywords.
* **Step 2:** KWE using [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) trained on a raw TeX corpus. This step yields some more general keywords relevant to the linguistic discipline.
* **Step 3:** KWE using TF-IDF trained on a detexed corpus. This step fills in some missing keywords that also appear in a reference corpus of 10,000 previously accepted keywords, ```keywordslist.txt```.

The number of steps can be controlled by (not) providing the relevant corpora during training. The result needs some manual correction and supplementation of relevant unigrams or trigrams.


# Installation
```cmd
pip3 install langscikw
```

Developed in Python 3.7.3 32-bit. Needs at least Python 3.7 and the following packages: jellyfish, joblib, networkx, scikit-learn, segtok, regex.

Download the langsci corpus files from [here](https://github.com/langsci/langscikw-corpus/releases/tag/v1.0.1) or use your own.


# Usage
## Command line 
The command line tool provides only a simple interface. If you'd like to customize the model parameters or the number of steps, please see below.

```cmd
langscikw inputfile [n] [corpus1] [corpus2] [keywordslist] [--silent]
```

The keywords are printed to the console and can be redirected to a text file.

### Arguments
* ```inputfile```: Path to a .txt or .tex file or directory from which to extract keywords.
* ```n```: *Optional* Number of keywords to extract. Defaults to 300.
* ```corpus1```: *Optional* Path to corpus for step 2, usually raw TeX files or a [joblib-compressed](https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html#joblib.dump) file. If not provided, looks for corpus_tex.gz in the current directory.
* ```corpus2```: *Optional* Path to corpus for step 3, usually detexed files or a [joblib-compressed](https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html#joblib.dump) file. If not provided, looks for corpus_detexed.gz in the current directory.
* ```keywordslist```: *Optional* Path to a list of gold keywords for step 3. A default list based on langsci publications is installed with the package.
* ```--silent```: *Optional* Only print the result to the console, no progress updates.

## KWE class
```python
import langscikw
input_path = "my_book"              # File/directory to extract keywords from
keywords_path = "keywords.txt"      # File to save keywords to

kwe = langscikw.KWE()
kwe.train("corpus_tex.gz", "corpus_detexed.gz")
kws = kwe.extract_keywords(input_path, n=300, dedup_lim=0.85)
for kw in kws:
    print(kw)                       # Keywords are alphabetically sorted strings
```

### Keyword arguments
* ```n```: *Optional* Number of keywords to extract. Defaults to 300.
* ```dedup_lim```: *Optional* Deduplication limit. Keywords that have a Jaro-Winkler Similarity of >dedup_lim are not added to the final list. Defaults to 0.85.


## Stand-alone models
The YAKE and TF-IDF models may also be used on their own. Please consult the docstrings for more information.

```python
import langscikw
yake = langscikw.yakemodel.YakeExtractor()      # -> extract_keywords()
tfidf = langscikw.tfidfmodel.TfidfExtractor()   # -> train() -> extract_keywords()
```
