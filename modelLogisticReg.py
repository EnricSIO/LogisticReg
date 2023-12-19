import pandas as pd
import time
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Charger les données
data = pd.read_csv('dataset.csv')

# Convertir la colonne cible en binaire (1 pour réussir, 0 pour échouer)
data['Curricular units 2nd sem (grade)'] = (data['Curricular units 2nd sem (grade)'] >= 10).astype(int)

# Sélection des caractéristiques
selected_features = [col for col in data.columns if col != 'Curricular units 2nd sem (grade)']
X = data[selected_features]
y = data['Curricular units 2nd sem (grade)']

# Séparer les caractéristiques catégorielles et numériques
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numeric_features = X.select_dtypes(include=['int', 'float']).columns.tolist()

# Préprocesseur pour gérer les caractéristiques catégorielles et numériques
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# Pipeline pour le prétraitement et le modèle
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', SVC())
])

# Paramètres pour GridSearchCV
param_grid = {
    'classifier__C': [0.1, 1, 10],  # Paramètres de régularisation
    'classifier__kernel': ['rbf', 'linear'],  # Types de noyau
    'classifier__gamma': ['scale', 'auto']  # Paramètre gamma
}

# Création du modèle GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Démarrer le chronomètre
start_time = time.time()

# Entraîner le modèle avec GridSearchCV
grid_search.fit(X_train, y_train)

# Arrêter le chronomètre
end_time = time.time()
elapsed_time = end_time - start_time
print("Temps d'exécution de GridSearchCV : {:.2f} secondes".format(elapsed_time))

# Meilleure combinaison de paramètres
print("Meilleurs paramètres : ", grid_search.best_params_)

# Faire des prédictions avec le meilleur modèle trouvé
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))

# Sélectionner un élève au hasard et ses caractéristiques
random_student = data.iloc[np.random.randint(len(data))]
random_student_features = random_student[selected_features]

# Préparer l'observation pour la prédiction
sample_for_prediction = pd.DataFrame([random_student_features])

# Faire une prédiction pour l'élève sélectionné
predicted_class = grid_search.predict(sample_for_prediction)
print("Classe prédite pour l'élève sélectionné :", predicted_class[0])

# Générer et afficher la matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Échec (0)', 'Réussite (1)'], 
            yticklabels=['Échec (0)', 'Réussite (1)'])
plt.title('Matrice de confusion pour le modèle SVM optimisé')
plt.ylabel('Vérité terrain')
plt.xlabel('Prédiction du modèle')
plt.show()