import os
import sys #manipulating function arguments

directory = sys.argv[1]
dataset = {}
folderSize = len(os.listdir(directory))
i = 0

while i != folderSize:
    line = os.listdir(directory)[i]
    if not str(line[11:-5]) in dataset:
        dataset[str(line[11:-5])] = [i,0]
    else:
        dataset[str(line[11:-5])][1] = i
    i+=1
for x, y in dataset.items():
  print(x, y)
