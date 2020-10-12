import cv2
import numpy as np



# loading pre trained data for detecting all frontal faces
pre_trained_data = cv2.CascadeClassifier(r"resources/haarcascade_frontalface_default.xml")



# importing the image of the face for detection
img = cv2.imread(r"resources/two faces.jpg")


# resizing the image, since the image is very big
resize_image = cv2.resize(img, (630,630))


# convert the image to greyscale
img_grey = cv2.cvtColor(resize_image,cv2.COLOR_BGR2GRAY)


# for getting the face coordinates
face_coordinates = pre_trained_data.detectMultiScale(image=resize_image)

print(f"face_coordinates_are: {face_coordinates}")

# specifying the entities of face_coordinates
print(f"x={face_coordinates[0][0]}\ny={face_coordinates[0][1]}\n"
      f"w={face_coordinates[0][2]}\nh={face_coordinates[0][3]}")

# drawing the rectangle around the face

for (x,y,w,h) in face_coordinates:
      """looping through faces for multiple face detection"""

      # creating a colourful rectangle around the face
      cv2.rectangle(img=resize_image,pt1=(x,y), pt2=(x+w, y+h),
                    color=(np.random.randint(1,255),np.random.randint(1,255),np.random.randint(1,255)),
                    thickness=2)


# img output
cv2.imshow("potraint", resize_image)
cv2.waitKey(delay=0)
