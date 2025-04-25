#Neccessary Imports

import os
import cv2
import pickle
import numpy as np
import face_recognition

#Save encodings
def saveEncodings(encs,names,fname="encodings.pickle"):

    data=[]
    d = [{"name": nm, "encoding": enc} for (nm, enc) in zip(names, encs)]
    data.extend(d)

    encodingsFile=fname
    
    # dump the facial encodings data to disk
    print("[INFO] serializing encodings...")
    f = open(encodingsFile, "wb")
    f.write(pickle.dumps(data))
    f.close()    

#Function to read encodings
def readEncodingsPickle(fname):

    data = pickle.loads(open(fname, "rb").read())
    data = np.array(data)
    encodings = [d["encoding"] for d in data]
    names=[d["name"] for d in data]
    return encodings,names

#Function to create encodings and get face locations
def createEncodings(image):

    #Find face locations for all faces in an image
    face_locations = face_recognition.face_locations(image)
    
    #Create encodings for all faces in an image
    known_encodings=face_recognition.face_encodings(image,known_face_locations=face_locations)
    return known_encodings,face_locations

#Function to compare encodings
def compareFaceEncodings(unknown_encoding,known_encodings,known_names):

    duplicateName=""
    distance=0.0
    matches = face_recognition.compare_faces(known_encodings, unknown_encoding,tolerance=0.5)
    face_distances = face_recognition.face_distance(known_encodings, unknown_encoding)
    best_match_index = np.argmin(face_distances)
    distance=face_distances[best_match_index]
    if matches[best_match_index]:
        acceptBool=True
        duplicateName=known_names[best_match_index]
    else:
        acceptBool=False
        duplicateName=""
    return acceptBool,duplicateName,distance

#Save Image to new directory
def saveImageToDirectory(image,name,imageName):
    path="./output/"+name
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)
    cv2.imwrite(path+"/"+imageName,image)
    


#Function for creating encodings for known people
def processKnownPeopleImages(path="./People/",saveLocation="./known_encodings.pickle"):
    known_encodings=[]
    known_names=[]
    for img in os.listdir(path):
        imgPath=path+img

        #Read image
        image=cv2.imread(imgPath)
        name=img.rsplit('.')[0]
        #Resize
        image=cv2.resize(image,(0,0),fx=0.2,fy=0.2,interpolation=cv2.INTER_LINEAR)
        
        #Get locations and encodings
        encs,locs=createEncodings(image)
        
        known_encodings.append(encs[0])
        known_names.append(name)
        
        for loc in locs:
            top, right, bottom, left=loc
            
        #Show Image
        cv2.rectangle(image,(left,top),(right,bottom),color=(255,0,0),thickness=2)
        cv2.imshow("Image",image)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
    saveEncodings(known_encodings,known_names,saveLocation)

#Function for processing dataset images
def processDatasetImages(path="./Dataset/",saveLocation="./dataset_encodings.pickle"):
    #Read pickle file for known people to compare faces from
    people_encodings,names=readEncodingsPickle("./known_encodings.pickle")
    
    for img in os.listdir(path):
        imgPath=path+img

        #Read image
        image=cv2.imread(imgPath)
        orig=image.copy()
        
        #Resize
        image=cv2.resize(image,(0,0),fx=0.2,fy=0.2,interpolation=cv2.INTER_LINEAR)
        
        #Get locations and encodings
        encs,locs=createEncodings(image)
        
        #Save image to a group image folder if more than one face is in image
        if len(locs)>1:
            saveImageToDirectory(orig,"Group",img)
        
        
        #Processing image for each face
        i=0
        knownFlag=0
        for loc in locs:
            top, right, bottom, left=loc
            unknown_encoding=encs[i]
            i+=1
            acceptBool,duplicateName,distance=compareFaceEncodings(unknown_encoding,people_encodings,names)
            if acceptBool:
                saveImageToDirectory(orig, duplicateName, img)
                knownFlag=1
        if knownFlag==1:
            print("Match Found")
        else:
            saveImageToDirectory(orig,"Unknown",img)
        
        #Show Image
        cv2.rectangle(image,(left,top),(right,bottom),color=(255,0,0),thickness=2)
        cv2.imshow("Image",image)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

        
def main():
    datasetPath="./Dataset/"
    peoplePath="./People/"
    processKnownPeopleImages(path=peoplePath)
    processDatasetImages(path=datasetPath)
    print("Completed")

if __name__=="__main__":
    main()
