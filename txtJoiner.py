import os
from datetime import date #for better organizing results
import sys #manipulating function arguments

with open("dataset_" + str(date.today()) + ".txt","w+") as dataset:        
    dir = sys.argv[1]
    files = os.listdir(dir)
    print('files',files)
    print('files type',type(files))
    files = [files.sort()]
    print('files type', type(files))
    print('files', files)

    for file in files :

            print('file',file)
            print('file',type(file))

            lines = open(os.path.join( dir, file) ,"r").read()
            dataset.write(lines+'\n')
