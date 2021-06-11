from imutils import paths
import face_recognition
import pickle
import cv2
import os
import sys
import numpy
from numpy import append

File_name = 'Testclip.mp4'
OUT_name = "Output.mp4"
encodingsFile = "encodings.pickle"
data = pickle.loads(open(encodingsFile, "rb").read())
encodingsFileN = "New_Encoding.pickle"

VP = cv2.VideoCapture(File_name)
if not VP.isOpened():
    print("Cannot open File")
    exit()

# Video_Details_capture
fps = VP.get(cv2.CAP_PROP_FPS)
W = int(VP.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(VP.get(cv2.CAP_PROP_FRAME_HEIGHT))
Frame_size = ((W, H))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
print("codec = ", fourcc, ", Fps = ", fps, " , Dimension = ", Frame_size)
OP = cv2.VideoWriter(OUT_name, fourcc, fps, Frame_size, isColor=True)


# Converting_To_GreyScale_and_Save
success, image = VP.read()
count = 0
Ident = []
Ident_image = []
SC ={}
UK = 0

while success:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb,model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    names =[]
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"],encoding)
        name = "Unknown"
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            name = max(counts, key=counts.get)
        names.append(name)
    for ((top, right, bottom, left), name) in zip(boxes, names):
        if(name == "Unknown"):
            print("Doing")
            encodings = numpy.array(encodings)
            try:
                encodings = encodings.reshape((128, ))
                print(encodings.shape)
                data["encodings"].append(encodings)
                data["names"].append("Unknown"+str(UK))
                UK += 1
                Ident.append("Unknown_"+str(UK))
                img = image[top:bottom,left:right]
                Ident_image.append(img)
                SC[name]=0
            except:
                print("Kuch NA Mila")
        
        else:
            try:
                SC[name]+=1
            except:
                SC[name]=0
            if "Unknown" in name:
                print("Frz")
                encodings = numpy.array(encodings)
                try:
                    encodings = encodings.reshape((128, ))
                    print(encodings.shape)
                    data["encodings"].append(encodings)
                    data["names"].append(name)
                except:
                    print("Kuch NA Mila")
            if name not in Ident:
                print("oh no Doing")
                SC[name]=0
                Ident.append(name)
                img = image[top:bottom,left:right]
                Ident_image.append(img)

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,0.75, (0, 255, 0), 2)

    #  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow("Image",image)
    OP.write(image)
    cv2.waitKey(1)
    success, image = VP.read()
    count += 1
    print(count)

VP.release()
OP.release()
cv2.destroyAllWindows()
j = 0

Pim ={}

for i in Ident_image:
    img = i
    pad = numpy.full((30,img.shape[1],3), [255,255,255], dtype=numpy.uint8)
    result = numpy.vstack((img,pad))
    X ,Y = result.shape[0],result.shape[1]
    print(X,Y)
    if "Unknown" in Ident[j]:
        img = cv2.putText(result, "Can you", (0,Y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) , 1, cv2.LINE_AA)
        img = cv2.putText(result, "Recognize ?", (1,Y+26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0) , 1, cv2.LINE_AA)
        cv2.namedWindow(Ident[j])
        cv2.moveWindow(Ident[j], 50,30)
        cv2.imshow(Ident[j],img)
        cv2.waitKey(0)
        K = int(input("Can You Recognize : 1:Yes  2:No\n"))
        if(K==1):
            val = input("Enter Label : ")
            for index, item in enumerate(data["names"]):
	            if item ==  Ident[j]:
		            data["names"][index] = val
            try:
                time = (1/fps)*(SC[Ident[j]]+1)
            except:
                time =  (1/fps)
            Ident[j] = val
            if Ident[j] not in Pim:
                L = []
                L.append(i)
                L.append(time)
                Pim[Ident[j]] = L
            else:
                Pim[Ident[j]][1] += time
    else:
        time = (1/fps)*(SC[Ident[j]]+1)
        img = cv2.putText(result, Ident[j], (0,Y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0) , 1, cv2.LINE_AA)
        img = cv2.putText(result,str(time)+" Sec" , (1,Y+26), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0) , 1, cv2.LINE_AA)
        time = (1/fps)*(SC[Ident[j]]+1)
        if Ident[j] not in Pim:
            L = []
            L.append(i)
            L.append(time)
            Pim[Ident[j]] = L
        else:
            Pim[Ident[j]][1] += time
        cv2.namedWindow(Ident[j])
        cv2.moveWindow(Ident[j], 50,30)
        cv2.imshow(Ident[j],img)
        cv2.waitKey(0)
    j+=1
    


print("Total Frames = ", count)

print("Saving Encodings...")
f = open(encodingsFileN, "wb")
f.write(pickle.dumps(data))
f.close() 


print("Saving graph.")
f = open("GRAPH.pickle", "wb")
f.write(pickle.dumps(Pim))
f.close() 

os.system('python Map.py')

print("Done")

cv2.destroyAllWindows()