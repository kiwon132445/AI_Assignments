#pip install sklearn
#pip install scikit-learn
#pip install pandas
#pip install spacy
#python -m spacy download en_core_web_lg
#pip install torch
#pip install tabulate
#pip install tqdm

import sys
import warnings
warnings.filterwarnings("ignore")

from Yelp_Manager import *

#main function
def main():
    if len(sys.argv) < 4:
        print("Not enough arguments\nGiven: " + len(sys.argv))
        return -1
    elif len(sys.argv) == 3:
        Yelp_Manager(filepath=sys.argv[1], model=sys.argv[2])
        yelp.run_model()
        return 0
    elif len(sys.argv) == 4:
        Yelp_Manager(filepath=sys.argv[1], model=sys.argv[2], nrows=sys.argv[3])
        yelp.run_model()
        return 0
    elif len(sys.argv) == 5:
        yelp = Yelp_Manager(filepath=sys.argv[1], model=sys.argv[2], nrows=sys.argv[3], sys.argv[4])
        yelp.run_model()
        return 0
    
    print("Incorrect arguments")
    return -1
        
    
if __name__ == "__main__":
    main()