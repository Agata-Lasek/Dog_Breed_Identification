import numpy as np
import pandas as pd
from PIL import Image, ImageOps           
import matplotlib.pyplot as plt
import os                                 
import random                             
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

def main():
    
    # dane wejściowe
    data = pd.read_csv('C:/Users/agata/Desktop/dog-breed-identification/labels.csv')
    test = pd.read_csv('C:/Users/agata/Desktop/dog-breed-identification/sample_submission.csv')
    print(data.head,"\n")                                                       
    print(data.shape,"\n")                                                             
    print(test.head,"\n")                                                               
    print(test.shape,"\n")    
    print("Liczba unikalnych nazw ras psów: ",data["breed"].nunique(),"\n")  
    print("Liczba brakujących danych: \n",data.isna().sum(),"\n")                     
    
    train = "C:/Users/agata/Desktop/train"                
    test="C:/Users/agata/Desktop/test"

    nrow = 5
    ncol = 5
    fig,ax=plt.subplots(nrow,ncol,figsize=(10,10))                  
    
    target_size=(224, 224)
    dog_images = []                                                 
    breeds = []                                                 
    

    for idx, (image_id, breed) in enumerate(data[["id", "breed"]].itertuples(index=False)):       
        image_path = os.path.join(train, f"{image_id}.jpg")                                     
        img = Image.open(image_path)                                                            
        img = img.resize(target_size)                                                          
        dog_images.append(img)
        breeds.append(breed)
        row = idx // ncol                                
        col = idx % ncol                               
        
    dog_images_stack = np.stack(dog_images)               
    breeds_stack = np.stack(breeds)
    
    for i in range(nrow * ncol):
            row = i // ncol                                
            col = i % ncol                               
            ax[row, col].imshow(dog_images_stack[i])       
            ax[row, col].set_title(breeds_stack[i])      
            ax[row, col].axis("off")                 

    plt.tight_layout()
    plt.show()                                             
    
    
    # 
    num_breed = data["breed"].nunique()                 
    batch_size = 32                                    
    encoder = LabelEncoder()                            
    encoded = encoder.fit_transform(data['breed'])     
    encoded_breed = to_categorical(encoded)            
    
    
   
    X = dog_images_stack                         
    y = encoded_breed                             
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)    
   
  
    model = Sequential()       
    
    model.add(Conv2D(32, kernel_size=(3, 3), activation ='relu', input_shape = (224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_breed, activation='softmax'))       
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  
    model.fit(X_train, y_train, batch_size, epochs=5)              


    # Load and preprocess test images
    dog_images_test = []

#    for idx, image_id in enumerate(test["id"]):
#        image_path = os.path.join(test_dir, f"{image_id}.jpg")
#        img = load_img(image_path, target_size=target_size)
#        img_array = img_to_array(img) / 255.0
#        dog_images_test.append(img_array)

#    dog_images_stack_test = np.stack(dog_images_test)

    # Make predictions on test data
#    predictions = model.predict(dog_images_stack_test)

    # Decode predictions
#    predicted_breeds_encoded = np.argmax(predictions, axis=1)
#    predicted_breeds = LabelEncoder().fit(breeds).inverse_transform(predicted_breeds_encoded)

    # Display 25 random test images with predicted breeds
#    random_indices = np.random.choice(len(dog_images_stack_test), 25, replace=False)
#    fig, ax = plt.subplots(5, 5, figsize=(10, 10))

#    for i, idx in enumerate(random_indices):
#        ax[i // 5, i % 5].imshow(dog_images_stack_test[idx])
#        ax[i // 5, i % 5].set_title(predicted_breeds[idx])
#        ax[i // 5, i % 5].axis("off")

#    plt.tight_layout()
#    plt.show()

if __name__ == '__main__':
    main()