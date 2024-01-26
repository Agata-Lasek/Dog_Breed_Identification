import numpy as np
import pandas as pd
from PIL import Image, ImageOps             #Wczytywanie obrazów bezpośrednio z plików
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import os                                   # manipulacja  ścieżkam
import random                               #do randomowych zdjec
#from sklearn.model_selection import StratifiedShuffleSplit
#from sklearn import preprocessing
from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
#from keras import optimizers
#from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')













def main():
    
    # dane wejściowe
    data = pd.read_csv('C:/Users/agata/Desktop/dog-breed-identification/labels.csv')
    print(data.head,"\n")                                                       #wyswietl kilka pierwszych linii
    print(data.shape,"\n")                                                      #tymczasowe - liczba danych
    print("Liczba unikalnych nazw ras psów: ",data["breed"].nunique(),"\n")  
    print("Liczba brakujących danych: \n",data.isna().sum(),"\n")               #mamy wszystkie więc można spokojnie działać
    
    train = "C:/Users/agata\Desktop/train"          # dane obrazy
    #test="C:/Users/agata\Desktop/test"
    
    # Liczba wierszy i kolumn w siatce
    nrow = 5
    ncol = 5
    fig,ax=plt.subplots(nrow,ncol,figsize=(10,10))                              # obiekt fig to cała figura, oraz tablica ax zawierającą podwykresy, z których każdy będzie używany do umieszczania jednego obrazu
    
    target_size=(224, 224)
    dog_images = []                                                                     # pusta lista do przechowywania obrazów psów
    breeds = []                                                                         # pusta lista do przechowywania ras psów
    for i, (img_id, breed) in enumerate(data[["id", "breed"]].values[: nrow * ncol]):   # Iteracja po pierwszych nrow*ncol wierszach z ramki danych data
        image_path = os.path.join(train, f"{img_id}.jpg")                               # Ścieżka do obrazu
        img = Image.open(image_path)                                                    # Wczytywanie obrazu za pomocą Pillow (PIL)
        img = img.resize(target_size)                                                   # Dostosuj rozmiar obrazu do wspólnego rozmiaru
        dog_images.append(img)
        breeds.append(breed)
        
    dog_images_stack = np.stack(dog_images)                                             # Przekształć listę obrazów, np.stack umożliwia łączenie sekwencji w nową oś
    breeds_stack = np.stack(breeds)
    
    # test wyswietlanie
    for i in range(25):
            row = i // ncol                                 # Oblicza aktualny wiersz
            col = i % ncol                                  # Oblicza aktualny kolumnę
            ax[row, col].imshow(dog_images_stack[i])        # Wyświetla obraz na odpowiednim podwykresie.
            ax[row, col].set_title(breeds_stack[i])         # Dodaje tytuł podwykresu, który jest nazwą rasy psa   
            ax[row, col].axis("off")                        # off oznaczenia osi, niekoniecznie - tylko estetyka

    plt.tight_layout()
    plt.show()
    
    
    
            
    




    # Inicjalizacja modelu
    model = Sequential()    



if __name__ == '__main__':
    main()