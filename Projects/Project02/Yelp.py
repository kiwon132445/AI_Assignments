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
        Yelp_Manager(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        return 0
    
    print("Incorrect arguments")
    return -1
        
    
if __name__ == "__main__":
    main()