import os
from datetime import date #for better organizing results
import sys #manipulating function arguments

with open("dataset_" + str(date.today()) + ".txt","w+") as dataset:        
    dir = sys.argv[1]
    files = os.listdir(dir)
    files = sorted(files)
    
    for file in files :

            print('3 file',file)
            print('3 file',type(file))

            lines = open(os.path.join( dir, file) ,"r").read()
            dataset.write(lines+'\n')
