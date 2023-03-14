import os
import json
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

class EnsambleModel:
    def __init__(self):
        # Define the individual models
        knn = KNeighborsClassifier()
        rf = RandomForestClassifier(max_depth=10)

        # Define the pipeline:
        # 1. minmax scaler
        # 2. feature extraction with pca for reducing the number of features 
        #    (you should test this apart for your specific dataset)
        # 3. ensamble model composed by random forest and support vector machines
        self.pipe = Pipeline([
            ('scaler', MinMaxScaler()),
            ("pca", PCA()), 
            ('ensemble', VotingClassifier(
                 estimators=[
                     ('knn', knn),
                     ('rf', rf),
                 ],
                 voting='soft')
             )
        ])

    def train(self, X, y, savedir='model_files'):
        
        # split in train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Define a minimal parameter grid
        # (the parameters are only for the ensamble components)
        param_grid = {
            'pca__n_components': [X.shape[1]//2], # reduce the features by half with PCA
            'ensemble__knn__n_neighbors': [3, 10], # knn neighbours
            'ensemble__rf__n_estimators': [5, 10], # estimators in random forest
        }

        # Create a GridSearchCV object
        grid = GridSearchCV(
            self.pipe,
            param_grid,
            cv=2, # the dataset is large so in this case only we perform a small cv
            n_jobs=-1,
            verbose=3,
            scoring='accuracy'
        )

        # Train the GridSearchCV
        grid.fit(X_train, y_train)

        # Print the best parameters and score
        print('Best parameters found by gridsearch:', grid.best_params_)
        print('Best cross validation score in gridsearch:', grid.best_score_)
        print('Score in test set:', grid.best_estimator_.score(X_test, y_test))

        # Save the best parameters to a JSON file
        os.makedirs(savedir, exist_ok=True)
        with open(os.path.join(savedir,'best_params.json'), 'w') as f:
            json.dump(grid.best_params_, f)

        # Save the best model to a binary file
        with open(os.path.join(savedir,'best_model.pkl'), 'wb') as f:
            pickle.dump(grid.best_estimator_, f)

    def predict(self, X):
        return self.pipe.predict(X)
    
    def predict_prob(self, X):
        return self.pipe.predict_proba(X)
            
    def load(self, loaddir='model_files'):
        # Load the saved model from binary file
        with open(os.path.join(loaddir,'best_model.pkl'), 'rb') as f:
            self.pipe = pickle.load(f)
            
