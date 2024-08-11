from keras.api.models import load_model
from PIL import Image
import numpy as np
import os

model=load_model('new_model.h5')
print("Loaded model from disk")

classs={0: 'Speed limit (5km/h)', 
            1: 'Speed limit (15km/h)',
            2: 'Speed limit (30km/h)', 
            3: 'Speed limit (40km/h)', 
            4: 'Speed limit (50km/h)', 
            5: 'Speed limit (60km/h)', 
            6: 'Speed limit (70km/h)', 
            7: 'speed limit (80km/h)', 
            8: 'Dont Go straight or left', 
            9: 'Unknown7', 10: 'Dont Go straight', 
            11: 'Dont Go Left', 
            12: 'Dont Go Left or Right', 
            13: 'Dont Go Right', 
            14: 'Dont overtake from Left', 
            15: 'No Uturn', 
            16: 'No Car', 
            17: 'No horn', 
            18: 'No entry', 
            19: 'No stopping', 
            20: 'Go straight or right', 
            21: 'Go straight', 
            22: 'Go Left', 
            23: 'Go Left or right',
            24: 'Go Right', 
            25: 'keep Left', 
            26: 'keep Right', 
            27: 'Roundabout mandatory', 
            28: 'watch out for cars', 
            29: 'Horn', 
            30: 'Bicycles crossing', 
            31: 'Uturn', 
            32: 'Road Divider', 
            33: 'Unknown6', 
            34: 'Danger Ahead', 
            35: 'Zebra Crossing', 
            36: 'Bicycles crossing', 
            37: 'Children crossing', 
            38: 'Dangerous curve to the left', 
            39: 'Dangerous curve to the right', 
            40: 'Unknown1', 41: 'Unknown2', 
            42: 'Unknown3', 43: 'Go right or straight', 
            44: 'Go left or straight', 45: 'Unknown4', 
            46: 'ZigZag Curve', 47: 'Train Crossing', 
            48: 'Under Construction', 
            49: 'Unknown5', 50: 'Fences', 
            51: 'Heavy Vehicle Accidents'
        }
print("Obtaining Images & its Labels ...........")

def load_and_process(image_path):
    image=Image.open(image_path)
    image=image.resize((30,30))
    image=image.convert('RGB')
    image=np.array(image)
    image=np.expand_dims(image,axis=0)
    return image

test_folder='./dataset/TEST/000_1_0004_1_j.png'
image=load_and_process(test_folder)
result=model.predict(image)[0]
class_pred=np.argmax(result)
sign=classs[class_pred]
print(f"Image :{test_folder} Predicted Traffic Sign :{sign}")
'''
for filename in os.listdir(test_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path=os.path.join(test_folder,filename)
        image=load_and_process(image_path)

        result=model.predict(image)[0]
        predicted_class_index=np.argmax(result)
        predicted_sign=classs[predicted_class_index]

        print(f"Image :{filename} Predicted Traffic Sign :{predicted_sign}")
'''
print("Testing Completed")