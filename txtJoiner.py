import os
from datetime import date #for better organizing results
import sys #manipulating function arguments

with open("dataset_" + str(date.today()) + ".txt","w+") as dataset:        
        dire = sys.argv[1]
    files = os.listdir(dir)
    files = sorted(files)
    
    for file in files:
        if file[-4:] == '.txt':
            print(file)
            print(type(file))

            lines = open(os.path.join( dire, file) ,"r").read()
            dataset.write(lines+'\n')
