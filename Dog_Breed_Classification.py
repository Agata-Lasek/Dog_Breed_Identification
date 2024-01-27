import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
#import matplotlib.image as mpimg
import os
import random
import warnings
warnings.filterwarnings('ignore')

#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
#from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator


def displayInputData(dogs: list, breeds: list) -> None:
    rows = 5
    columns = 5

    # obiekt fig to cała figura, oraz tablica ax zawierającą podwykresy, z których każdy będzie używany do umieszczania jednego obrazu
    fig, ax=plt.subplots(rows, columns, figsize=(10,10))    

    for row in range(rows):
         for col in range(columns):
            # Wyświetla obraz na odpowiednim podwykresie
            ax[row, col].imshow(dogs[row * rows + col])  

             # Dodaje tytuł podwykresu, który jest nazwą rasy psa       
            ax[row, col].set_title(breeds[row * rows + col])  

             # off oznaczenia osi, niekoniecznie - tylko estetyka   
            ax[row, col].axis("off")                       

    plt.tight_layout()
    plt.show()


def main() -> None:
    # dane wejściowe
    data = pd.read_csv("data/labels.csv")

    print(data.head(),"\n")
    print(data.shape,"\n")

    print("Liczba unikalnych nazw ras psów: ", data["breed"].nunique(),"\n")  
            
    dog_images = [] # pusta lista do przechowywania obrazów psów
    breeds = []     # pusta lista do przechowywania ras psów

    target_size = (224, 224)
    
    for indx, (image_id, breed) in enumerate(data[["id", "breed"]].itertuples(index=False)):
        # Ścieżka do obrazu
        image_path = os.path.join("data/train", f"{image_id}.jpg")

        # wczytaj rozmiar za pomoca pillow i dostosuj rozmiar obrazu
        img = Image.open(image_path)
        img = img.resize(target_size)

        dog_images.append(img)
        breeds.append(breed)

    # Przekształć listę obrazów, np.stack umożliwia łączenie sekwencji w nową oś
    dog_images_stack = np.stack(dog_images)                
    breeds_stack = np.stack(breeds)

    displayInputData(dog_images[:25], breeds[:25])   
    
    # Ustawienia
    num_breed = data["breed"].nunique()                 # Ustala docelową liczbę ras psów
    batch_size = 32                                     # 32 oznacza, że będą przetwarzane 32 obrazy jednocześnie podczas trenowania modelu (wydaje sie optymalne ale mozna zmieniac)
    encoder = LabelEncoder()                            # przekształcanie etykiet breed na liczby Przyporządkowuje unikalną liczbe całkowitą dla każdej unikalnej etykiety
    encoded = encoder.fit_transform(data['breed'])      # przypisywane numeryczne wartości do każdej kategorii breed
    encoded_breed = to_categorical(encoded)             # zakodowane etykiety w formie jako wektor zer i jednej jedynki (ułatwia pracę z danymi kategorycznymi w kontekście modeli uczenia maszynowego)
    
    
    # Przygotowanie danych do treningu modelu
    X = dog_images_stack                          # tablicą NumPy zawierającą zmniejszone obrazy psów
    y = encoded_breed                             # zmienna, która przechowuje przekształcone etykiety
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30, stratify=y)      # oznacza, że 20% danych zostanie użyte jako zbiór testowy w paczkach po 30 obrazow, X i y, gdzie X zawiera obrazy, a y zawiera odpowiadające zakodowane etykiety
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

    # Convolutional Neural Network
    model = Sequential()        # Inicjalizacja modelu
    
    model.add(Conv2D(32, kernel_size=(3, 3), activation ='relu', input_shape = (224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_breed, activation='softmax'))       # warstwa gęsta, num_breed określa liczbę neuronów w warstwie-  jest równoważne liczbie klas (120)
    
    learning_rate = 1e-2
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])  # Kompilacja modelu, optymalizator, loss-funkcja straty określa, jak bardzo przewidywane wyjścia modelu różnią się od rzeczywistych etykiet
    model.fit(train_generator,steps_per_epoch= X_train.shape[0] // batch_size, epochs=5,
                 validation_data= test_generator,
                 validation_steps= X_test.shape[0] // batch_size)
    
    model.summary()
    
    #y_pred = model.predict(test)                                                          #do uzyskania prognoz na podstawie danych testowych test
    
    
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