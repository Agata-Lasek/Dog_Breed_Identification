import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import os
import random
import warnings
warnings.filterwarnings('ignore')

#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical 
from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
#from keras import optimizers
#from keras.preprocessing.image import ImageDataGenerator


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
    X_train = dog_images_stack                          # zmienna, która przechowuje obrazy
    y_train = encoded_breed                             # zmienna, która przechowuje przekształcone etykiety
    




    # Inicjalizacja modelu
    model = Sequential()    


if __name__ == '__main__':
    main()