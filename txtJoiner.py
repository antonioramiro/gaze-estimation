import os
from datetime import date #for better organizing results
import sys #manipulating function arguments

with open("dataset_" + str(date.today()) + ".txt","w+") as dataset:        
    dir = sys.argv[1]
    files = os.listdir(dir).sort()
    for file in files:
            lines = open( os.path.join( dir, file ) ,"r").read()
            dataset.write(lines+'\n')
