import cv2
import sys
import numpy as np
from joblib import load
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
from openpose import pyopenpose as op

##vizzy Ready: a partir de uma imagem, usa o OpenPose para extrair as posições 2D dos keypoints da cabeça e a partir daí, estima para onde a pessoa está a olhar

if sys.argv[0][-4]==".": #Diferencia imagens .png (onde o -4 é o .) de imagens provenientes da camara do vizzy, que vêm como objetos (???)
    img = cv2.read(sys.argv[0]) #Se for um png ou jpg, converte em imagem cv2
else:
    img = sys.argv[0] #caso contrário, mantem o formato
     
h, w = img.shape[:2]  #aquisição das dimensões da imagem
h = int(h) 
w= int(w)

if int(h/w) != int(16/9): #Como o algoritmo foi treinado com imagens 16:9, é necessário converter as que não o sejam para 16:9
    blank_image = np.zeros(shape=[1280, 720, 3], dtype=np.uint8) #Duvido que esta parte funcione, mas é suposto pegar numa imagem branca com a porporção certa e dar overlay da imagem em questão, ficando assim com barras laterais brancas
    img = cv2.addWeighted(img,1,blank_image,0,0) #na linha de cima foi criada a imagem branca, aqui é feita a adição, com 100% de opacidade

def master(imagem):
    params = dict() #load do openpose 
    params["model_folder"] = "/home/sims/repositories/openpose/models/"
      
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    
    frame = cv2.resize(imagem,(1280 ,720)) #resize para 1290 por 720: se estiver em 16:9 é tranquilo, caso contrário fica distorcido
    datum = op.Datum() #estas linhas processam a imagem
    imageToProcess = frame 
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum]) 
    ok = True 
    
    try:
        keypoints = (datum.poseKeypoints[0]) #output do open pose, caso existam pessoas no enquadramento
    except(IndexError):
        ok = False
        print('No people found') #output se não forem detetadas pessoas
    
    if ok: #se tiverem sido detetadas pessoas, cria a lista com os keypoints
        falhado = 0 #medidor de erro
        
        fim = [keypoints[0][0],keypoints[0][1],\
               keypoints[17][0],keypoints[17][1],\
               keypoints[18][0],keypoints[18][1],\
               keypoints[15][0],keypoints[15][1],\
               keypoints[16][0],keypoints[16][1]] #lista com os keypoints
        
        fim = [-1 if x==0 else x for x in fim] #substituir os 0 por -1, a assinalar o erro
        for x in fim:
            if x == -1:
                falhado += 1 #por cada 0 que existisse nos keypoints (=posição não detetada), o contador aumenta um

        if (sys.argv[1]) == "y": #visualização da imagem
            cv2.imshow('image',frame)
       
        cv2.waitKey(50)

        if falhado < 3: #se o openpose consegui detetar todas as posições menos uma (duas coordenadas), então:
            clf2 = load('classificador1.joblib') #load do classificador criado anteriormente
            focus = (clf2.predict(np.array([fim]))) #usar o classificador para detetar a que desaseis'ante pertence o vetor de coordenadas fim
    return focus #output do quadrante
  
def look(n): #definição das coordenadas para onde é suposto o vizzy olhar, fiz em estilo grelha, ou seja
    x,y=0    #primeiro vai procurar qual o y a que corresponde o quadrante (entre 4 opções) e analogamente para o x 
    if n in [0,2,8,10]:
        x = int((w*1)/8)
        
    elif n in [1,3,9,11]:
        x = int((w*3)/8)
        
    elif n in [4,6,12,14]:
        x = int((w*5)/8)
        
    elif n in [5,7,13,15]:
        x = int((w*7)/8)
        
    if n in [0,1,4,5]:
        y = int((h*1)/8)
        
    elif n in [2,3,6,7]:
        y = int((h*3)/8)
        
    elif n in [8,9,12,13]:
        y = int((h*5)/8)
        
    elif n in [10,11,14,15]:
        y = int((h*7)/8)
        
    z = 22 #falta estimar o z, baseado no x e y
                
    return x,y,z

look(master(img)) #executa a função de olhar para as coordenadas correspondentes ao output da master, quando lhe é dada a imagem


    






