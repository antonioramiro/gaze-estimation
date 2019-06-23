# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:06:08 2019

@author: toshiba
"""

import os

    
with open("teste.txt","w+") as final:        
    dir = "/home/sims/Dropbox/CoLabs-VizzyBrain/teste"
    for file in os.listdir( dir ):
          if file.endswith( ".txt" ):
              lines=open( os.path.join( dir, file ) ,"r").read()
              final.write(lines+'\n')
              