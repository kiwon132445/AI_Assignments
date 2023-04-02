#pip install sklearn
#pip install scikit-learn
#pip install pandas
#pip install spacy
#python -m spacy download en_core_web_lg
#pip install torch

#unused so far
#pip install -U spacy[transformers]
#python -m spacy download en_core_web_trf

#pip install tweet-preprocessor
#pip install transformers

#pip install gensim
#pip install -U pip setuptools wheel
#python -m spacy download en_core_web_sm
#python -m spacy download en_core_web_md

import sys
import warnings
warnings.filterwarnings("ignore")

from Yelp_Manager import *

#main function
def main():
    if len(sys.argv) < 4:
        print("Not enough arguments\nGiven: " + len(sys.argv))
        return -1
    elif len(sys.argv) == 4:
        Yelp_Manager(sys.argv[1], sys.argv[2], sys.argv[3])
        return 0
    elif len(sys.argv) == 5:
        if sys.argv[4] == "True":
            Yelp_Manager(sys.argv[1], sys.argv[2], sys.argv[3], True)
        elif sys.argv[4] == "False":
            Yelp_Manager(sys.argv[1], sys.argv[2], sys.argv[3], False)
        return 0
    
    print("Incorrect arguments")
    return -1
        
    
if __name__ == "__main__":
    main()