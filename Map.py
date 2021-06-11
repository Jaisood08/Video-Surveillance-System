import cv2
import numpy
import pickle
import sys

data = pickle.loads(open("GRAPH.pickle", "rb").read())
# print(data)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import matplotlib.cbook as cbook
from io import BytesIO
colors = ['crimson', 'dodgerblue', 'teal', 'limegreen', 'gold','blue','green','red','cyan','magenta','yellow','black']
values = []
for key in data:
    print(key)
    values.append(data[key][1])

labels = list(data.keys())
colors = colors[0:len(labels)]

print(values)
height = 0.9

plt.barh(y=labels, width=values, height=height, color=colors, align='center')

i = 0
for key in data:
    value = data[key][1]
    content = "temp.png"
    cv2.imwrite(content,data[key][0])
    im = mpimg.imread(content)
    print(type(im))
    print(im.shape)
    plt.imshow(data[key][0], extent=[value - 0.01, value - 0.05, i - height + 0.5 , i + height -0.5], aspect='auto', zorder=2)
    extent=[value - 8, value - 2, i - height / 2, i + height / 2]
    print(extent)
    i+=1

plt.xlim(0, max(values) * 1.05)
plt.ylim(-0.5, len(labels) - 0.5)
plt.tight_layout()
plt.xlabel('Seconds')
plt.ylabel('Actors')

plt.savefig("Actor_map.png", facecolor='w', bbox_inches="tight",
            pad_inches=0.3, transparent=True)
print("Saved")
plt.show()
sys.exit()