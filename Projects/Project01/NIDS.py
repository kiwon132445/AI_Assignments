#pip install sklearn
#pip install scikit-learn
#pip install pandas

import sys
import warnings
warnings.filterwarnings("ignore")

from NIDS_Manager import *

#main function
def main():
    if len(sys.argv) < 4:
        print("Not enough arguments")
        return -1
    elif len(sys.argv) == 4:
        csv_name = sys.argv[1]
        c_method = sys.argv[2]
        task = sys.argv[3]
        
        ndis = NIDS_Manager(csv_name, c_method, task)
        return 0
    elif len(sys.argv) == 5:
        csv_name = sys.argv[1]
        c_method = sys.argv[2]
        task = sys.argv[3]
        model_name = sys.argv[4]
        
        ndis = NIDS_Manager(csv_name, c_method, task, model_name)
        return 0
    
    print("Incorrect arguments")
    return -1
        
    
if __name__ == "__main__":
    main()