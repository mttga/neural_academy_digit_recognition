# Riconoscimento Cifre Numeriche con un Ensamble Model

## Modello

Questo repository contiene una classe Python chiamata "EnsembleModel" che implementa una pipeline di machine learning utilizzando un modello di ensemble composto da classificatori K-Nearest Neighbors (KNN) e Random Forest (RF). La pipeline include la normalizzazione dei dati, la riduzione della dimensionalità mediante Principal Component Analysis (PCA) e il modello di ensemble stesso. La classe ha tre metodi principali: "train" per addestrare la pipeline e salvare il miglior modello e i relativi parametri, "predict" per effettuare previsioni su nuovi dati e "load" per caricare un modello pre-addestrato dal disco. Il metodo di addestramento utilizza GridSearchCV per effettuare una ricerca degli iperparametri sui classificatori KNN e RF, e i migliori parametri sono salvati in un file JSON. Il miglior modello è salvato in un file binario utilizzando il modulo pickle. 

La classe è progettata per essere flessibile, in modo che gli utenti possano modificare gli iperparametri della pipeline o utilizzare un dataset diverso cambiando i dati in ingresso al metodo train.

### Web Application

![Digit Recognition](https://github.com/nai-kon/cnn-digit-recognition-webapp/raw/master/demo.gif)

- Flask per web framework
- d3.js per disegnare i grafici

## Requirements
- `pip install -r requirements.txt`

## Uso

- #### WebApp
  `python3 app.py`   
  ->poi accedi a `localhost:5000`
  
- #### Addestra la pipeline (con gridsearch)
  `python3 train.py`

