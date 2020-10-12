import cv2
import numpy as np



# loading pre trained data for detecting all frontal faces
pre_trained_data = cv2.CascadeClassifier(r"resources/haarcascade_frontalface_default.xml")

# importing the image of the face for detection
img = cv2.imread("insert your image here")

# convert the image to greyscale
img_grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# for getting the face coordinates
face_coordinates = pre_trained_data.detectMultiScale(image= img_grey)

print(f"face_coordinates_are: {face_coordinates}")

# specifing the entities of face_cordinates
print(f"x={face_coordinates[0][0]}\ny={face_coordinates[0][1]}\n"
      f"w={face_coordinates[0][2]}\nh={face_coordinates[0][3]}")

# drawing the rectangle around the face
# pt1=(x,y), pt2=(w+x,h+y)
cv2.rectangle(img, face_coordinates, (0, 255, 0), 5)


# img output
cv2.imshow("potraint", img)
cv2.waitKey(delay=0)
