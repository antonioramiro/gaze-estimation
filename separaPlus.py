import random
import shutil
import os
import site
import sys


def separaPlus():   
    dir="lixo"
    list = os.listdir(dir)
    j=False
    i=0
    
    for i in range(11643):
        file=random.choice(list)
        while j==False:
            if file.endswith('.txt'):
                j=True
            else:
                 file=random.choice(list)
        srcpath0=os.path.join(dir, file)
        shutil.move(srcpath0, "treino")
        shutil.move(srcpath0[:-4]+".png","treino")
        list = os.listdir(dir)
        j=False
        
    for i in range(2910):
        file=random.choice(list)
        while j==False:
            if file.endswith('.txt'):
                j=True
            else:
                file=random.choice(list)
        srcpath1=os.path.join(dir, file)
        shutil.move(srcpath1, "teste")
        shutil.move(srcpath1[:-4]+".png","teste")
        list = os.listdir(dir)
        j=False
separaPlus()