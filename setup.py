from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='langscikw',
    version='0.0.1',
    description='Keyword extraction from linguistic publications',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://pypi.python.org/pypi/langscikw',
    author='Noel Simmel',
    author_email='noelsimmel@protonmail.com',
    keywords='langsci keywordextraction indexing keywords naturallanguageprocessing nlp',
    license="LGPLv3",

    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: End Users/Desktop',
        # Topics adapted from YAKE
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: Indexing',
        'Topic :: Text Processing :: Linguistic',
        'Natural Language :: English',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
    ],

    packages=find_packages(),

    python_requires='>=3.7, <4',
    install_requires=['scikit-learn==0.24.2', 'regex==2021.4.4', 'jellyfish', 'joblib', 'networkx==2.5.1', 'segtok'],

    package_data={
        'langscikw': ['corpora/*.txt'],
    },

    entry_points={
        'console_scripts': [
            'langscikw=langscikw.cli:main',
        ],
    },

    project_urls={
        'Source': 'https://github.com/langsci/langscikw',
        'Bug Reports': 'https://github.com/langsci/langscikw/issues',
        'Language Science Press': 'https://langsci-press.org/',
    },
)
