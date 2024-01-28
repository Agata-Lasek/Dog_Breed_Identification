import numpy as np
import pandas as pd
from PIL import Image, ImageOps           
import matplotlib.pyplot as plt
import tensorflow as tf
import os                                 
import random                             
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.callbacks import  ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

def load_and_resize_images(image_folder, target_size):
    images = []
    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        img = Image.open(image_path)
        img = img.resize(target_size)
        images.append(img)
    return images

def display_images_with_breeds(images, breeds, nrow, ncol):
    fig, ax = plt.subplots(nrow, ncol, figsize=(10, 10))

    for i in range(nrow * ncol):
        row = i // ncol
        col = i % ncol
        ax[row, col].imshow(images[i])
        ax[row, col].set_title(breeds[i])
        ax[row, col].axis("off")

    plt.tight_layout()
    plt.show()

def main():
    

    data = pd.read_csv('C:/Users/agata/Desktop/dog-breed-identification/labels.csv')
    print(data.head,"\n")                                                              
    print(data.shape,"\n")                                                         
    print("Liczba unikalnych nazw ras psów: ",data["breed"].nunique(),"\n")  
    print("Liczba brakujących danych: \n",data.isna().sum(),"\n")                   
    
    train = "C:/Users/agata/Desktop/train"          
    random_img="C:/Users/agata/Desktop/random_img"
    
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
    
    
    # Ustawienia
    num_breed = data["breed"].nunique()                
    batch_size = 32                                  
    encoder = LabelEncoder()                          
    encoded = encoder.fit_transform(data['breed'])    
    encoded_breed = to_categorical(encoded)        
    
    

    X = dog_images_stack                       
    y = encoded_breed                          
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30, stratify=y)  
    y_train
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    train_generator = train_datagen.flow(X_train,y_train, batch_size)

    test_datagen = ImageDataGenerator()
    
    test_generator = test_datagen.flow(X_test,y_test, batch_size)


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
    
    learning_rate = 1e-2
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']) 
    model.fit(train_generator,steps_per_epoch= X_train.shape[0] // batch_size, epochs=1,
                 validation_data= test_generator,
                 validation_steps= X_test.shape[0] // batch_size)
    
    model.summary()


    csv_file_path = "C:/Users/agata/Desktop/identyficated.csv"

    dog_images_test = load_and_resize_images(random_img, target_size)
    breeds_empty = [""] * len(dog_images_test)

    df_identified = pd.DataFrame({"id": os.listdir(random_img), "breed": breeds_empty})

    df_identified.to_csv(csv_file_path, index=False)

    random_test_images = random.sample(os.listdir(random_img), 20)
    dog_images_test = load_and_resize_images(random_img, target_size)

    if dog_images_test:
        dog_images_stack_test = np.stack(dog_images_test)
        predictions = model.predict(dog_images_stack_test)

        predicted_breeds_encoded = np.argmax(predictions, axis=1)
        predicted_breeds = encoder.inverse_transform(predicted_breeds_encoded)

        for image_file, predicted_breed in zip(random_test_images, predicted_breeds):
            df_identified.loc[df_identified["id"] == image_file, "breed"] = predicted_breed

        df_identified.to_csv(csv_file_path, index=False)

        display_images_with_breeds(dog_images_test, predicted_breeds, nrow=5, ncol=5)
    else:
        print("Lista obrazów testowych jest pusta.")

if __name__ == '__main__':
    main()