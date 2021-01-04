import tensorflow.keras
from PIL import Image, ImageOps
import urllib.request
import numpy as np
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('image_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def predict(image_file):
  image = Image.open(image_file)

  size = (224, 224)
  image = ImageOps.fit(image, size, Image.ANTIALIAS)

  image_array = np.asarray(image)
  image.show()

  normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
  data[0] = normalized_image_array

  prediction = model.predict(data)
  return prediction
  
  
  def get_names():
  f = open('image_labels.txt')
  names = f.read()
  f.close()
  names = names.split("\n")
  return names
  
  def process_prediction(image):
  filename = "image.jpg"
  urllib.request.urlretrieve(image, filename)

  arr = predict(filename)
  arr = arr.tolist()

  arr = str(arr)
  arr = arr.replace("[", "")
  arr = arr.replace("]", "")
  arr = arr.split(", ")

  newList = []

  for num in arr:
    num = float(num)
    newList.append(num)

  names = get_names()

  for num in range(len(newList)):
    if newList[num] == max(newList):
      maxNum = num
    
  newPrediction = names[maxNum]
  confidence = str(max(newList))

  return newPrediction, confidence
  
  
  while True:
  image = input("Web>>")
  if image == "quit":
    break
  else:
    try:
      name, confidence = process_prediction(image)
      print(name + " - " + confidence)
    except Exception as e:
      print(e)
