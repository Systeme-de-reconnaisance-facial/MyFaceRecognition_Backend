import base64
from cv2 import RETR_LIST
from flask import Flask, jsonify
import cv2
import numpy as np
import json
import os.path
import werkzeug
from PIL import Image
from numpy import asarray
import pandas as pd
import csv
from flask_cors import CORS
# import request
from flask import request,Response
#from flask_sqlalchemy import SQLAlchemy
import sqlite3
from datetime import date
from deepface import DeepFace
import json
import os
#from skimage.exposure import is_low_contrast
from skimage.io import imread, imshow, imsave
from skimage import*
from flask import g
import threading
from flask_caching import Cache

#import torch 
#import torchvision.transforms.functional as con
#import matplotlib.pyplot as plt


#function to detect face from image
def faceDetect(img,filename):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        #save file
        path_file = ("static/User."+filename)
        cv2.imwrite(path_file, gray[y:y+h,x:x+w])
        #response
        resp = "Image Uploaded Successfully"
    except:
        resp = "This image does not have a face"
    print(resp)
    #response
    return resp 


#function to get images and ids
def getImagesAndLabels(path):
	imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
	faces = []
	Ids = []
	for imagePath in imagePaths:
		pilImage = Image.open(imagePath).convert('L')
		imageNp = np.array(pilImage, 'uint8')
		Id = int(os.path.split(imagePath)[-1].split(".")[1])
		faces.append(imageNp)
		Ids.append(Id)
	return np.array(Ids),faces


#function for train images
def TrainImages(): 
    try:
        #recognizer=cv2.face.createEigenFaceRecognizer()
        #recognizer=cv2.face.createFisherFaceRecognizer()
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        path="static"
        Ids,faces = getImagesAndLabels(path)
        recognizer.train(faces,Ids)
        recognizer.save("trainingData.yml")
        resp="training image successfully"
    except:
        resp = "This image does not have a face, so cannot train image"
    print(resp)
        
 
        
        
#API   
app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
CORS(app)
 
@app.route("/")
def showHomePage():
    #resp = "This is home page "
    resp = []
    try:
        connect = sqlite3.connect('mycampusface.db')
        cursor = connect.cursor()
        cursor.execute('''SELECT * FROM ETUDIANT''')

        data = cursor.fetchall()
        connect.commit()
        connect.close() 
        resp=data
        
        with open("imagesTorecognize/null.1.jpg", "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read())
    except:
       resp = "empty"
    print(resp)    
    return ""
    


connect = sqlite3.connect('mycampusface.db')
connect.execute(
    '''CREATE TABLE IF NOT EXISTS ETUDIANT( id int NOT NULL, username varchar(100),\
    usersurname varchar(100) NOT NULL, usermatricule char(7) NOT NULL, userfiliere varchar(50) NOT NULL, userniveau varchar(5) NOT NULL, usertransaction text, usertranche varchar(10), userprice varchar(20))''')
connect.commit()
connect.close()

conn = sqlite3.connect('mycampusface.db')
conn.execute(
    '''CREATE TABLE IF NOT EXISTS RECONNU( ids int NOT NULL, nom varchar(100) NOT NULL,\
    surname varchar(100) NOT NULL, matricule char(7) NOT NULL, filiere varchar(50) NOT NULL, niveau varchar(5) NOT NULL, num_transaction text, tranche varchar(10), price varchar(20), annee date)''')
conn.commit()
conn.close()


DATABASE = 'mycampusface.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db


@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()
        
        
@app.route("/getuserid")
def getUserId():
    try:
        connect = sqlite3.connect('mycampusface.db')
        cursor = connect.cursor()
        cursor.execute('''SELECT * FROM ETUDIANT''')
        data = cursor.fetchall()
        
        if len(data)==0:
            return "1"
        else:
            num = len(data)+1
            return str(num)  
    except:
        return "0" 

lock = threading.Lock()  
        
    
@app.route("/web/sample", methods=["POST"])
def debugweb():
    data = request.get_json()
    sample = data["sample"]

    text = sample.split(',')
    try:
        connect = sqlite3.connect('mycampusface.db')
        cursor = connect.cursor()
        cursor.execute("""INSERT INTO ETUDIANT(id,username,usersurname,usermatricule,userfiliere,userniveau,usertransaction,usertranche,userprice) VALUES (?,?,?,?,?,?,?,?,?)""",
                       (text[0],text[1],text[2],text[3],text[4],text[5],text[6],text[7],text[8]))
        connect.commit()
        connect.close()
    except:
        print("error")
        
    return "received" 
 
@app.route("/sample", methods=["POST"])
def debug():
    texte = request.form["sample"]
    text = texte.split(',')
    try:
        connect = sqlite3.connect('mycampusface.db')
        cursor = connect.cursor()
        cursor.execute("""INSERT INTO ETUDIANT(id,username,usersurname,usermatricule,userfiliere,userniveau,usertransaction,usertranche,userprice) VALUES (?,?,?,?,?,?,?,?,?)""",
                       (text[0],text[1],text[2],text[3],text[4],text[5],text[6],text[7],text[8]))
        connect.commit()
        connect.close()
    except:
        print("error")
        
    return "received" 


@app.route("/api/upload", methods=["GET","POST"])
def upload():
    #retrieve image from client 
    imagefile = request.files["image"]
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    print("\nReceived image File name : " + imagefile.filename)
    imagefile.save("ClientImages/"+filename)
    img = asarray(Image.open("ClientImages/"+filename))
    #process image
    #input_img = imread("ClientImages/"+filename)
    #image_bright = exposure.adjust_gamma(input_img, gamma=0.5,gain=1)
    #imsave("ClientImages/"+filename,image_bright)
    img_processed = faceDetect(img,filename)
    #TrainImages()
    #Response
    return img_processed


@app.route("/api/recognize_faces", methods=["GET","POST"])
def detect_faces():      
    #receive image of faces
    our_image = request.files["image"]
    filename = werkzeug.utils.secure_filename(our_image.filename)
    #save that image to directory and transform that image to numpy array
    print("\nReceived image File name : " + our_image.filename)
    our_image.save("imagesTorecognize/"+filename)
    # input_img = imread("imagesTorecognize/"+filename)
    # image_bright = exposure.adjust_gamma(input_img, gamma=0.5,gain=1)
    image = Image.open("imagesTorecognize/"+filename)
    
    #Start recognition
    path="ClientImages"
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)] 
    for imagePath in imagePaths:
        print(imagePath) 
        try:
            result = DeepFace.verify(img1_path = image, img2_path = imagePath, distance_metric="euclidean_l2")
            print(result)
            if(result['verified']==True and result['distance'] <= 0.5):
                Id = int(os.path.split(imagePath)[-1].split(".")[0])
                print(str(Id))
                connect = sqlite3.connect('mycampusface.db')
                cursors = connect.cursor()
                cursors.execute('''SELECT * FROM ETUDIANT WHERE id={}'''.format(Id))
                datas = cursors.fetchall()
                for row in datas:
                    resp = str(row[0])+','+str(row[1])+','+str(row[2])+','+str(row[3])+','+str(row[4])+','+str(row[5])+','+str(row[6])+','+str(row[7])+','+str(row[8])
                    with sqlite3.connect('mycampusface.db') as user:
                        cursors = user.cursor()
                        cursors.execute("""INSERT INTO RECONNU(ids,nom,surname,matricule,filiere,niveau,num_transaction,tranche,price,annee) VALUES (?,?,?,?,?,?,?,?,?,?)""",(str(row[0]),str(row[1]),str(row[2]),str(row[3]),str(row[4]),str(row[5]),str(row[6]),str(row[7]),str(row[8]),date.today()))
                    user.commit()
                    user.close()
                break
            else:
                resp = "Unknown"
        except :
            resp="Image with any face or image of poor quality"
    print (resp)    
    #Response
    return resp

"""    try:
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  
    for imagePath in imagePaths:
        #rec=cv2.face.createEigenFaceRecognizer()
        #rec=cv2.face.createFisherFaceRecognizer()
        rec=cv2.face.LBPHFaceRecognizer_create()
        rec.read("trainingData.yml")
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            # To draw a rectangle in a face
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)
            id, uncertainty = rec.predict(gray[y:y+h, x:x+w])
            print(id,uncertainty)
            
            connect = sqlite3.connect('mycampusface.db')
            cursors = connect.cursor()
            cursors.execute('SELECT * FROM ETUDIANT WHERE id={}'.format(id))
            datas = cursors.fetchall()
        
            if (uncertainty<= 100):
                for row in datas:
                    resp = str(row[0])+','+str(row[1])+','+str(row[2])+','+str(row[3])+','+str(row[4])+','+str(row[5])+','+str(row[6])+','+str(row[7])+','+str(row[8])
                    with sqlite3.connect('mycampusface.db') as user:
                        cursors = user.cursor()
                        cursors.execute("INSERT INTO RECONNU(id,username,usersurname,usermatricule,userfiliere,userniveau,usertransaction,usertranche,userprice,annee) VALUES (?,?,?,?,?,?,?,?,?,?)",(str(row[0]),str(row[1]),str(row[2]),str(row[3]),str(row[4]),str(row[5]),str(row[6]),str(row[7]),str(row[8]),date.today()))
                    user.commit()
            else:
                resp = "Unknown"     
    except:
        resp
    print (resp)""" 
   


@app.route("/api/users_recognize")
def getUsers():
    #resp = "This is home page "
    resp = []
    
    con = get_db()
    cursor = con.cursor()
    # Verrouille la connexion à la base de données
    cursor.execute('BEGIN')
    try:    
        cursor.execute("""SELECT DISTINCT ids,nom,surname,matricule,filiere,niveau,num_transaction,tranche,price,annee FROM RECONNU;""")
        data = cursor.fetchall()
        for row in data:
            resp.append(str(str(row[0])+'/'+str(row[1])+'/'+str(row[2])+'/'+str(row[3])+'/'+str(row[4])+'/'+str(row[5])+'/'+str(row[6])+'/'+str(row[7])+'/'+str(row[8])+'/'+str(row[9])))
            # print(resp)  
    except:
        resp = "empty"
        # En cas d'erreur, annule les opérations en cours
        con.rollback()
        raise
    finally:
        # Déverrouille la connexion à la base de données
        cursor.execute('COMMIT')
    print(resp)    
    return resp  #json.dumps(data)



@app.route("/update", methods=["POST"])
def update():
    text = request.form["sample"]
    # Obtenir l'adresse IP du client
    user_ip = request.remote_addr
    print(f'Traitement de la requête de {user_ip}')
    # Créer un nouv eau thread pour le client
    thread = threading.Thread(target=process_request3, args=(text,))
    thread.start()
    resp = process_request3(text)
    return resp


def process_request3(texte):
    lock.acquire()
    resp=""
    text = texte.split(',')
    trans=text[2]
    try:
        db = sqlite3.connect('mycampusface.db')
        cursor = db.cursor()
        cursor.execute("UPDATE ETUDIANT SET usertransaction = ?, usertranche = ?, userprice = ? WHERE id = ?",(trans,text[3],text[4],text[0]))
        db.commit()
        db.close()
        resp = "User informations update successfully"
    finally:
        lock.release()
    return resp

 
if __name__ == "__main__":
  app.run(host="0.0.0.0",port=500,threaded=True)