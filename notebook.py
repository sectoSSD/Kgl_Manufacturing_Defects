#coding:utf-8

"""
Bonjour à tous,

[FR]

Dans ce script nous allons ...

Un problème de classification binaire avec un petit volume de données qui appelle une solution
efficace mais simple. Nous allons donc implementer un SVC et un KNN qui ont l'avantage d'être rapide
et souvent efficaces. Reste à observer les données en détailles pour trouver les preprocessings qui
s'imposent

[ENG]

---------------------------------------------------------------------------------
  1 - Préliminaires / preliminary
---------------------------------------------------------------------------------

"""

# Bibliothèque ... / libraries ...
import math
import tkinter as tk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from time import perf_counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import  RandomOverSampler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# constante
VERBOSE = True
TARGET = 'DefectStatus'
TARGET_LABELS = ['Defective', 'Compliant']

# Jeu de données / data ...
raw_data = pd.read_csv('/home/sectossd/repositories/kaggle/Kgl_Manufacturing_Defects/raw_data.csv')

# Outils / tools

def format_target(target) :
    try :
        if target.shape[1] == 1 : target = np.array(target).ravel()
        elif target.shape[1] > 1 : target = np.argmax(target, axis= 1)
    except IndexError: 
        pass
    return target

# Interface d'affichage tkinter / Tkinter GUI

"""
Ce script n'est pas rédigé depuis un utilitaire (Ipython, Jupyter, VScode, SPyder ...) mais depuis le terminal avec VIM. Il faut donc quelques
fonction d'affichage réalisé avec tkinter.
"""

class Display(tk.Tk) :

    TITLE_HEIGHT = 24

    def __init__(self, title, width, height, *args, **kwargs) :

        tk.Tk.__init__(self, *args, **kwargs)
        self.width=width
        self.wm_title(title)
        self.wm_geometry(f"{width}x{height}")
    
    def add_title(self, title) :

        canvas = tk.Canvas(self, width=self.width, height=self.TITLE_HEIGHT)
        canvas.create_text( self.width/2, self.TITLE_HEIGHT/2, text=title, fill='white')
        canvas.pack()

    def add_text(self, title, text, height) : 
        
        self.add_title(title)
        text_height = height - self.TITLE_HEIGHT
        canvas = tk.Canvas(self, width=self.width, height=text_height, highlightthickness=0)
        canvas.create_text(self.width/2, text_height/2, text=text, fill='white')
        canvas.pack()
    
    add_array  = add_text

    def add_chart(self, title, figure, height, *args, **kwargs) :
        
        self.add_title(title)
        canvas = FigureCanvasTkAgg(figure, master = self)
        canvas.draw() 
        canvas.get_tk_widget().pack(pady=20)

def display_scores(title, y_test, y_pred, model_best_parameters, start_time, sample_best_parameters=None) :

    # check target format 
    y_test = format_target(y_test)
    y_pred = format_target(y_pred) 
    
    # time
    time = math.ceil((perf_counter() - start_time)/60)

    # Tkinter Frame
    frame=Display(f" Scores : {title} ({time} minutes)", 600, 750)

    # Classifaction report
    classification_report_ = classification_report(y_test, y_pred, target_names = TARGET_LABELS)
    frame.add_text('Classification report', classification_report_, 150)
    
    # Best parameters
    if sample_best_parameters : frame.add_text('Sample parameters', sample_best_parameters, 75)
    frame.add_text('Model parameters', model_best_parameters, 75)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(8,8), layout='constrained')
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax = ax, display_labels = TARGET_LABELS)
    frame.add_chart('Confusion Matrix', fig, 450)

    frame.mainloop()
"""
#######################

def correlation_matrix(dataframe) :

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Data and plot
    matrix = dataframe.corr()
    fig, ax = plt.subplots(figsize=(8,8), layout='constrained')
    sns.heatmap(matrix, ax=ax, annot=True, fmt=".2f", linewidth=.5)
    
    display_plot(fig, 'correlation matrix', 1200, 1000)

    ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(raw_y_test, y_pred, labels = np.unique(y_pred)),
                           display_labels = np.unique(y_pred)).plot(ax=ax[3])
    ax[3].set_title("FINAL MODEL")
    plt.show()

##########################
----------------------------------
2 - Variables / features 
----------------------------------


[FR-ENG]

lignes/rows : 3229

colone/columns : 18

0  - index : int
1  - ProductionVolume     / Volume de production               : integer
2  - ProductionCost       / Coût de production                 : float
3  - SupplierQuality      / indice de qualité des fournisseurs : float
4  - DeliveryDelay         / débit, taux de livraison           : float
5  - DefectRate           / taux de défaut                     : float
6  - QualityScore         / score de qualité                   : float
7  - MaintenanceHours     / nombre d'heures de maintenance     : integer
8  - DowntimePercentage   / pourcentage de temps d'arrêt       : float
9  - InventoryTurnoVer    / indice de rotation des stocks      : float
10 - StockoutRate         / indice de rupture de stock         : float
11 - WorkerProductivity   / productivité des travailleurs      : float
12 - SafetyIncidents      / incident de sécurité               : integer
13 - EnergyConsumption    / consomation énergétique            : float
14 - EnergyEfficiency     / efficacité énergétique             : float
15 - AdditiveProcessTime  / Temps de production supplémantaire : float
16 - AdditiveMaterialCost / Coût du matériaux supplémentaire   : float
17 - DefectStatus         / Statut de defaut                   : boolean  ==> TARGET

reparition de la cible / target balance :
DefectStatus = 0 : 517
DefectStatus = 1 : 2723

-----------------------------------------
3 - Méthode / method
-----------------------------------------
            
[FR]

Il y a trop de variable comparativement à la taille du jeu de donnée. Plutôt que d'appliquer un preprocessing a des variables qu'on supprimera
probablement, on tâchera d'appliquer au plus vite notre SVM. Celui-ci permet en effet d'observer l'importance de chaque variable (feature importance) 
dansl'apprentissage. Une fois les variables significative selectionnées, et en fonction du score, on observera plus en details les variables restantes
pour voir si des traitement sont nécéssaires.

Si le resultat ne sont pas satisgfaisant ...

Commençons déjà par vérifier l'ingtégrité de la base de données, corrigeons le déséquilibre de la variable cible et )mettons à mettosn à l'echelle (scaling) les données.

[ENG]

-------------------------------------------------
4 - Préparation des données / Preprocessiong 
------------------------------------------------
"""


# Données manquante / missing data

#display_dataframe(raw_data.iloc[0:5,:].to_string(), 'raw data', 1800, 200)
# display_dataframe(raw_data.describe(), 'describe', width=1000, height=200)
# NE MARCHE PAS DESCRIBE S'ADAPTE A LA TAILLE DU TERMINALE ET PAS A CELLE DE LA FENETRE
    
"""
[FR]
Pas de données manquante

[ENG]
"""

# Jeu de test et jeu d'entrainement / Test set and train set

def split(data) :

    if VERBOSE : print("","SPLITING", sep='\n')

    # between features and target
    if VERBOSE : print('', 'between features and target', sep='\n')
    y = data[f"{TARGET}"].to_frame('target')
    X = data.drop(f"{TARGET}", axis=1)
    if VERBOSE : print("y :", y.shape,"X :", X.shape)

    # between train and test set
    if VERBOSE : print('', 'between train and test set', sep='\n')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=69)
    if VERBOSE : print("X_train / X_test / y_train / y_test")
    if VERBOSE : print(X_train.shape, "/", X_test.shape, "/", y_train.shape, "/", y_test.shape )
    
    # convert to numpy with the good shape
    if VERBOSE : print('', 'change Y_train shape', f"y_train.shape before : {y_train.shape}", sep='\n')
    y_train = format_target(y_train)
    if VERBOSE : print(f"y_train.shape after : {y_train.shape}")

    if VERBOSE : print(".... DONE", "", sep="\n")    
    
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split(raw_data)

"""

[FR]

Dans cette étape préliminaire nous utiliserons un standard scaller qui est plus robuste contre les données abérantes. La mise a l'echelle serra 
inclus dans le pipeline  du SVM. On developpera ce premier model de telle façon qu'on pourra le réutiliser comme lui.

[ENG]

-----------------------------------
5 - Premier Essai / first try
-----------------------------------
"""

def config_pipeline(scaler, sampler, model) :    
    
    # set pipeline
    steps = list()
    if scaler != None : steps.append(("scaler", scaler))
    if sampler != None : steps.append(("sampler", sampler))
    if model != None : steps.append(("model", model))
    
    return Pipeline(steps=steps)

def grid_search(X_train, y_train, scaler, sampler, model, scorer, parameters) :
    
    if VERBOSE : print('', 'GRID_SEARCH_CV', sep='\n')

    # set pipeline
    if VERBOSE : print('', 'set pipeline', sep='\n')
    pipeline = config_pipeline(scaler, sampler, model) 
    if VERBOSE : print( f"pipeline : {pipeline}")

    # set grid search
    if VERBOSE : print('', 'set grid search', sep='\n')
    grid_pipeline = GridSearchCV(pipeline, parameters, cv=4, scoring=scorer, n_jobs=-1, verbose=1)
    grid_pipeline.fit(X_train, y_train)

    # research of best parameters
    if VERBOSE : print('', 'research of best parameters', sep='\n')
    gridsearch_params = grid_pipeline.best_params_
    sampl_best_params = set_best_parameters(gridsearch_params, "sampler") if sampler else None
    if sampler and VERBOSE : print(f"sampler best parameters : {sampl_best_params}")
    model_best_params = set_best_parameters(gridsearch_params, "model" )
    if VERBOSE : print(f"model best parameters : {model_best_params}")

    if VERBOSE : print('', '... DONE', sep='\n')
    
    return sampl_best_params, model_best_params

def set_best_parameters(gridsearch_params, tag) :
    
    # recast the parameters
    delete = tag + "__"
    keys = [key.replace(delete,'') for key in list(gridsearch_params.keys()) if tag in key]
    values = [val for key,val in gridsearch_params.items() if tag in key]
    best_params = {key : val for key, val in zip(keys, values)}
    
    return best_params

def fitting_and_prediction(X_train, y_train, X_test, scaler, sampler, sampl_best_params, model, model_best_params) :
    
    if VERBOSE : print('', 'FITTING AND PREDICTION', sep='\n')

    # config sampler and model
    if VERBOSE : print('', 'config sampler and model', sep='\n')
    if sampler : sampler.set_params(**sampl_best_params)
    if sampler and VERBOSE : print(f"sample : {sampler}")
    model.set_params(**model_best_params)
    if VERBOSE : print(f"model : {model}")

    # config fitting pipeline
    if VERBOSE : print('', 'set fitting pipeline', sep='\n')
    pipeline = config_pipeline(scaler, sampler, model)
    if VERBOSE : print( f"{pipeline}")

    # fit and predict
    if VERBOSE : print('', 'fitting and prediction', sep='\n')
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    classes = list(map(str, pipeline.classes_))
    
    if VERBOSE : print('', '... DONE', sep='\n')

    return y_pred, classes

def simple_svm(X_train, X_test, y_train, y_test) :
    title = 'SVM'
    start_time = perf_counter()
    print('*'*15, title, '*'*15 )
    
    parameters = {'sampler__sampling_strategy' : [{0: i} for i in [500, 750, 1000]],
                  'sampler__random_state' : [69],
                  'model__C': [0.1, 1, 10, 100, 1000], 
                  'model__gamma': [1, 0.1, 0.01], 
                  'model__kernel': ["rbf"], 
                  'model__class_weight' : ["balanced", None]}

    scaler = RobustScaler()
    sampler = RandomOverSampler()
    model = SVC()
    scorer = 'balanced_accuracy'
    
    sampl_best_params, model_best_params  = grid_search(X_train, y_train, scaler, sampler, model, scorer, parameters)

    y_pred, classes = fitting_and_prediction(X_train, y_train, X_test, scaler, sampler, sampl_best_params, model, model_best_params) 
    
    print('')
    print('*'*15, f"{title}", '*'*15)
    
    display_scores(title, y_test, y_pred, model_best_params, start_time, sampl_best_params)
    
    return y_pred
    
simple_svm_y_pred = simple_svm(X_train, X_test, y_train, y_test)

"""
--------------------
SUIVANT
-------------------

def plot_distribution(data, targert) :

    import matplotlib.pyplot as plt

    # creer un sns.pairplot from scratch avec matplotlib (mais commencer par limiter le nombre des variables) 
    
    import seaborn as sns
    
    # Seaborn pairplot
    fig, ax = plt.subplots(figsize=(12,8), layout='constrained')
    pairplot = sns.pairplot(data, hue=TARGET, palette="hls", height=0.7)

    # change axes labels guidances :
    for axe in pairplot.axes.flatten() :
        # rotate x axis labels
        axe.set_xlabel(axe.get_xlabel(), rotation='vertical', stretch='extra-condensed')
        # rotate y axis labels
        axe.set_ylabel(axe.get_ylabel(), rotation='horizontal', stretch='extra-condensed')
        # set y labels alignment
        axe.yaxis.get_label().set_horizontalalignment('right')

    fig = pairplot.fig

    display_plot(fig, 'Seaborn Pairplot', 1600, 1000)
"""
