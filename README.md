# Dog Breed Classification

Celem było zbudowanie modelu klasyfikującego rasę psa na podstawie zdjęcia. Do realizacji zadania wykorzystano głęboką sieć neuronową CNN, język Python oraz biblioteki TensorFlow/Keras.

## Opis projektu
Model klasyfikuje zdjęcia psów do jednej z 30 wybranych ras. Dane wejściowe pochodzą z
https://www.kaggle.com/code/dansbecker/tensorflow-programming-daily/input

## Proces obejmuje:

- wczytanie i przetwarzanie danych obrazowych,
- przygotowanie zbioru uczącego i testowego,
- augmentację danych,
- budowę modelu CNN od podstaw,
- trening i testowanie modelu,
- zapis przewidywanych wyników oraz wizualizację rezultatów.

## Dane wejściowe
Obrazy: Ponad 10 000 zdjęć psów (użyto 30 ras, ok. 2500 obrazów)

Etykiety: plik CSV (corrected_labels.csv) zawierający pary id + breed

Rozmiar zdjęć został ujednolicony do 224x224 pikseli. Obrazy zostały przefiltrowane tak, aby uwzględniać tylko wybrane 30 ras.


## Wymagania
- Python 3.11+
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- PIL
- Scikit-learn

## Uruchamianie

Należy uruchomić skrypt:
python Dog_Breed_Classification.py



