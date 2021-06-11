from imutils import paths
import face_recognition
import pickle
import cv2
import os

#Essential
DataFolder = 'dataset'
encodingsFile = 'encodings.pickle'
detection_method = 'cnn'


print("Starting Encoding")
imagePaths = list(paths.list_images(DataFolder))
knownEncodings = []
knownNames = []
name = ' '
F = 1
F_C = 0

#Calculating Encoding
for (i, imagePath) in enumerate(imagePaths):    
	name1 = imagePath.split(os.path.sep)[-2]
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	if (name1!=name):
		name = name1
		list = os.listdir("Dta/"+name)
		F_C = len(list)
		F = 1
	print(name," ",F,"/",F_C)
	F+=1
	boxes = face_recognition.face_locations(rgb,model=detection_method)
	encodings = face_recognition.face_encodings(rgb, boxes)
	for encoding in encodings:
		knownEncodings.append(encoding)
		knownNames.append(name)
        
#Saving Encodings
print("Saving Encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(encodingsFile, "wb")
f.write(pickle.dumps(data))
f.close()        

print("Done")