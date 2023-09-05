
import cv2
import imutils
import math


# Initializing the HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
   
# Reading the Image
image = cv2.imread('C:/Users/vaish/OneDrive/Desktop/aeolus/humans.jpg')
   
print(image.shape)
# Resizing the Image
image = imutils.resize(image,width=min(400, image.shape[1]))
   
# Detecting all the regions in the Image that has a pedestrians inside it
(regions, _) = hog.detectMultiScale(image, winStride=(4, 4), padding=(4, 4), scale=1.04)
print(regions)

(x_,y_,w_,h_)=regions[0]
cv2.rectangle(image, (x_, y_), (x_ + w_, y_ + h_), (0, 0, 255), 2)

# Looping through each image
for (x, y, w, h) in regions[1:]:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    dist=math.sqrt((x+w/2-x_-w_/2)*(x+w/2-x_-w_/2)+(y+h/2-y_-h_/2)*(y+h/2-y_-h_/2))
    cv2.line(image,(int(x+w/2),int(y+h/2)),(int(x_+w_/2),int(y_+h_/2)),(255,255,255),thickness=2)
    cv2.putText(image,str(int(dist)),(int((x+x_+w_/2+w/2)/2),int((y+y_+h_/2+h/2)/2)),cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,0.5,(255,255,255),thickness=2)
    x_=x
    y_=y
    w_=w
    h_=h

  
# Showing the output Image
cv2.imshow("Image", image)

cv2.waitKey(0)   
cv2.destroyAllWindows()
