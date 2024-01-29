import numpy as np
import pandas as pd
from PIL import Image, ImageOps           
import matplotlib.pyplot as plt
import tensorflow as tf
import os                                 
import random 
import shutil                            
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.callbacks import  ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')


def main():
    

    data = pd.read_csv('C:/Users/agata/Desktop/dog-breed-identification/corrected_labels.csv')
    print(data.head,"\n")                                                              
    print(data.shape,"\n")                                                         
    print("Liczba unikalnych nazw ras psów: ",data["breed"].nunique(),"\n")  
    print("Liczba brakujących danych: \n",data.isna().sum(),"\n")                   
    
    train = "C:/Users/agata/Desktop/train"        
    
    nrow = 5
    ncol = 5
    #fig,ax=plt.subplots(nrow,ncol,figsize=(10,10))                  
    
    target_size=(224, 224)
    dog_images = []                                              
    breeds = []                                                     
    
    
    breeds_list = ['american_staffordshire_terrier', 'australian_terrier', 'basset', 'blenheim_spaniel', 'cardigan', 'chihuahua', 'dingo', 'doberman', 'eskimo_dog', 'giant_schnauzer', 'german_shepherd', 'golden_retriever', 'irish_terrier', 'irish_water_spaniel', 'italian_greyhound', 'japanese_spaniel', 'kelpie', 'komondor', 'kuvasz', 'leonberg', 'mexican_hairless', 'norfolk_terrier', 'papillon', 'redbone', 'rottweiler', 'shih-tzu', 'siberian_husky', 'sussex_spaniel', 'toy_terrier', 'vizsla']

    for idx, (image_id, breed) in enumerate(data[["id", "breed"]].itertuples(index=False)):  
        image_path = os.path.join(train, f"{image_id}.jpg")       
        
        if breed in breeds_list:
            img = Image.open(image_path)                                                        
            img = img.resize(target_size)                                                           
            dog_images.append(img)
            breeds.append(breed)
        else:
            continue  
                           
    
    
    def copy_images_to_folder(csv_file_path, source_folder, destination_folder):

        # Create destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Copy images to the destination folder
        for image_name in data['id']:
            source_path = os.path.join(source_folder, f"{image_name}.jpg")
            destination_path = os.path.join(destination_folder, f"{image_name}.jpg")

            # Check if the source file exists before copying
            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
                print(f"Copied: {source_path} to {destination_path}")
            else:
                print(f"Warning: {source_path} not found.")

    # Specify paths
    csv_file_path = 'C:/Users/agata/Desktop/dog-breed-identification/corrected_labels.csv'
    source_folder = 'C:/Users/agata/Desktop/train'
    destination_folder = 'C:/Users/agata/Desktop/selected_images'

    # Copy images to the destination folder based on the CSV file
    copy_images_to_folder(csv_file_path, source_folder, destination_folder)  
    
    
        
    dog_images_stack = np.stack(dog_images)              
    breeds_stack = np.stack(breeds)
    

    random_indices = random.sample(range(len(dog_images_stack)), 25)

    fig, ax = plt.subplots(nrow, ncol, figsize=(10, 10))

    for i, index in enumerate(random_indices):
        row = i // ncol
        col = i % ncol
        ax[row, col].imshow(dog_images_stack[index])
        ax[row, col].set_title(breeds_stack[index])
        ax[row, col].axis("off")

    plt.tight_layout()
    plt.show()                                    
    
    
    # Ustawienia
    num_breed = data["breed"].nunique()                
    batch_size = 8                                  
    encoder = LabelEncoder()                          
    encoded = encoder.fit_transform(data['breed'])    
    encoded_breed = to_categorical(encoded)        
    
    

    X = dog_images_stack                       
    y = encoded_breed                          
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20, stratify=y)  
    y_train
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
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
    
    model.add(Conv2D(4, kernel_size=(3, 3), activation ='relu', input_shape = (224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(4, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(8, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(34, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    

    model.add(Flatten())
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    
    model.add(Dense(num_breed, activation='softmax'))    
    
    learning_rate = 1e-3
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy']) 
    model.fit(train_generator,steps_per_epoch= X_train.shape[0] // batch_size, epochs=150,
                 validation_data= test_generator,
                 validation_steps= X_test.shape[0] // batch_size)
    
    model.summary()
    
    #model.save('dog_bread_classifier.model')
    #model = model.load_model('dog_bread_classifier.model')


    def load_and_resize_images(image_folder, target_size):
        images = []
        for image_file in os.listdir(image_folder):
            image_path = os.path.join(image_folder, image_file)
            img = Image.open(image_path)
            img = img.resize(target_size)
            images.append(np.array(img) / 255.0)  # Normalize test data
        return images

    csv_file_path = "C:/Users/agata/Desktop/identyficated.csv"

    dog_images_test = load_and_resize_images(destination_folder, target_size)
    breeds_empty = [""] * len(dog_images_test)

    df_identified = pd.DataFrame({"id": os.listdir(destination_folder), "breed": breeds_empty})

    df_identified.to_csv(csv_file_path, index=False)

    random_test_images = random.sample(os.listdir(destination_folder), 20)
    dog_images_test = load_and_resize_images(destination_folder, target_size)

    
    if dog_images_test:
        dog_images_stack_test = np.stack(dog_images_test)
        predictions = model.predict(dog_images_stack_test)

        predicted_breeds_encoded = np.argmax(predictions, axis=1)
        predicted_breeds = encoder.inverse_transform(predicted_breeds_encoded)

        for image_file, predicted_breed in zip(random_test_images, predicted_breeds):
            df_identified.loc[df_identified["id"] == image_file, "breed"] = predicted_breed

        df_identified.to_csv(csv_file_path, index=False)
        
        
        def display_images_with_breeds(images, breeds, nrow, ncol):
            fig, ax = plt.subplots(nrow, ncol, figsize=(10, 10))


            for i in range(nrow * ncol):
                row = i // ncol
                col = i % ncol
                ax[row, col].imshow(dog_images_stack_test[i])
                ax[row, col].set_title(predicted_breeds[i])
                ax[row, col].axis("off")

            plt.tight_layout()
            plt.show()
        
    

        # Display images with predicted breeds
        display_images_with_breeds([Image.fromarray(img.astype(np.uint8)) for img in dog_images_stack_test], predicted_breeds, nrow=5, ncol=5)
    else:
        print("Lista obrazów testowych jest pusta.")

if __name__ == '__main__':
    main()