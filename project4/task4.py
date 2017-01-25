#Imports


import os
import zipfile
import json
from scipy import sparse
import numpy as np



#Dataset path

datadirectory="C:\\Users\\ahmad\\Desktop\\mmds\\lastfm_test.zip"
dataset = zipfile.ZipFile(datadirectory)



#global variables initialization


songId=0
t=0
g=50
beta=0.2
categories =["Technical Death Metal"]
S =[]
id2songJsonMap = []
id2songIdMap = {}
fhat=[]
n=5 #top songs to report
#limitingCounter = 1000
powerIterations = 100



#Loading data


for i in dataset.namelist():
    if  i.endswith(".json"): #songId<limitingCounter and 
        f = json.loads(dataset.read(i).decode("utf-8"))
        if  f["track_id"] not in id2songIdMap.keys():
            id2songIdMap[f["track_id"]] = songId
            tempcategories = set(categories)#will keep removing tags from it
            newtags=[]
            for tag in f["tags"]:#looping over tags
                if int(tag[1]) >g:#filters out those <g
                    newtags.append(tag)#keeping >g
                    if tag[0] in tempcategories:#if it is in the tempcategories
                        tempcategories.remove(tag[0])#removing it
            f["tags"] = newtags
            id2songJsonMap.append(f)
            #songTags= [t[0] for t in f["tags"]]
            #if len(list(filter(lambda x: x in categories,songTags )) ) == len(categories):
            if len(tempcategories)==0:#meaning we encountered all of the tags
                S.append(songId)
                fhat.append(1)
            else:
                fhat.append(0)
            songId+=1 



#initializing the teleportation probablilities



fhat=(1/len(S))*np.array(fhat)




#Building the glorious Adjacency matrix


col=[]
row=[]
data=[]
for i in range( len(id2songJsonMap)):
    for s in id2songJsonMap[i]["similars"]:
        if s[0] in id2songIdMap.keys() and s[1]>=t:
            col.append(i)
            row.append(id2songIdMap[s[0]])
            data.append(1)
    
    
    

        
 A = sparse.csr_matrix((data, (row, col)),shape=(104212, 104212))





 #Randomly initializing r


r =np.random.rand(A.shape[0])
r=r/np.sum(r)


#Power Iterations

for i in range(powerIterations):
    r=A.multiply(beta).dot(r)+fhat*(1-beta)    



#Reporting the top n songs

topIndecs = r.argsort()[-n:][::-1]


for item in topIndecs:
    print (id2songJsonMap[item]["track_id"])


