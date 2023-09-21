import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression, LogisticRegression 
import seaborn as sns
import numpy as npy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score

#raw load github
dataDefaulUrl = 'https://raw.githubusercontent.com/pipe2015/Projects_unad/main/data_csv';

class loadDataGraphics:
    def __init__(self):
        self.select_data = None;
        self.loadIndex = -1;
        self.list_csv = {
            'regresion_lineal_data': [
                dataDefaulUrl + '/regresion-lineal.csv' 
            ],
            'regresion_logistica_data': [
                dataDefaulUrl + '/regresion-logistica.csv'
            ],
            'tree_desicion_data': [
                dataDefaulUrl + '/tree-desicion.data'
            ]
        };
    
    def get_data_csv(self, type = "regresion_lineal_data", index = 0, **kwargs): # default parameter type => '', index => 
        self.loadIndex = index;
        try:
            if type in self.list_csv:
                print(self.list_csv.get(type)[index]);
                self.select_data = pd.read_csv(self.list_csv.get(type)[index], **kwargs);
                return self.select_data;
            raise Exception('list csv', 'Not found')
        except Exception as inst:
            print('EEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE')
            print(type(inst))
        except:
            print(f"Load csv not fount #{type} in #{index}");
            quit();
            
    def getValues(self) : return self.select_data.values; #Numpy representation object
    def getlength(self) : return self.select_data.size; #numero de rows object file
    def getRows(self, x = 10): return self.select_data.head(x); #default 10 rows counts
    def getDatacolums(self, search): return self.select_data.get(search);
    def loadGraphicsScatter(self, isShow, **kwargs): 
        self.select_data.plot.scatter(**kwargs);
        if isShow: plot.show();
    def viewData(self): print(self.select_data.to_json);

class LinearProgresionData(loadDataGraphics): 
    #init 
    def __init__(self):
        super().__init__()
        self.select_model = 'regresion_lineal_data';
        
    def start(self, index = 0):
        data_select = self.get_data_csv(self.select_model, index, sep=','); 
        
        # print Data view 
        self.viewData();
        
        # view Data Scatter 
        self.loadGraphicsScatter(True, x="metro", y='precio');
        
        #Agrego los datos en un array
        years_x = data_select['metro'].values.reshape((-1, 1));
        price_y = data_select['precio'];
        #add create model
        model = LinearRegression().fit(years_x, price_y);
        
        print('interseccion (b)', model.intercept_)
        print('Pendiente (m)', model.coef_)
        
        input_list = [[5],[7],[10],[12],[20], [25]]
        predicciones = model.predict(input_list)
        print(predicciones)
        
        # view load Data Scatter 
        self.loadGraphicsScatter(False, x="metro", y="precio", label='Datos originales');
        plot.scatter(input_list, predicciones, color='red')
        plot.plot(input_list, predicciones, color='black', label='Línea de regresión')
        plot.xlabel('metro')
        plot.ylabel('precio')
        plot.legend()
        plot.show()
    
    
class LogisticProgresionData(loadDataGraphics): 
    #init 
    def __init__(self):
        super().__init__()
        self.select_model = 'regresion_logistica_data';
    
    def start(self, index = 0): 
        data_select = self.get_data_csv(self.select_model, index, sep=',');
        
        print(self.getRows());
        
        data_comuns = self.getDatacolums(['BMI','currentSmoker']);
        
        print(data_comuns.head()) # default 5 rows 
        print(data_comuns.plot.scatter(x='BMI',y='currentSmoker'))
        plot.show(); # open gui
        
        # delete rows diff empy values 
        model_clear_data  = data_select.dropna();
        print(model_clear_data.head())

        # Agrego los datos en un array
        x_model_cleaned = np.array(model_clear_data['BMI']).reshape((-1, 1))
        y_model_cleaned = np.array(model_clear_data['currentSmoker'])

        # create model
        model = LogisticRegression().fit(x_model_cleaned, y_model_cleaned);
        
        print('interseccion (b)', model.intercept_)
        print('Pendiente (m)', model.coef_)
        
        line_rect = self.get_line_rect(model_clear_data);
        
        # show graphics Data recta
        data_select.plot.scatter(x='BMI',y='currentSmoker')
        plot.plot(line_rect['x'], line_rect['y'], 'red')
        plot.ylim(0, data_select['currentSmoker'].max() * 1.1)
        # plt.grid()
        plot.show()
        
    def get_line_rect(self, model_clear, w = -0.08, b = 2.08):
        x = np.linspace(0, model_clear['BMI'].max(), 100);
        y = 1 / ( 1 + np.exp(-(w * x + b)));
        return {'x' : x, 'y' : y};



class TreeDesicionData(loadDataGraphics): 
    #init 
    def __init__(self):
        super().__init__()
        self.select_model = 'tree_desicion_data';
    
    def start(self, index = 0): 
        header = None;
        names =  names = ["Class","Alcohol","Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium", "Total_phenols", "Flavonoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280/OD315_of_diluted wines", "Proline"];
        data_select = self.get_data_csv(self.select_model, index, header = header, names = names);
        
        print(self.getRows());
        
        print(data_select.describe());
        
        plot.hist(data_select.Class)
        plot.show(); # open gui
        
        predictors_col = ["Alcohol","Malic_acid", "Ash", "Alcalinity_of_ash", "Magnesium", "Total_phenols", "Flavonoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280/OD315_of_diluted wines", "Proline"]
        target_col = ["Class"]
        
        #Se asignan según su clase respectiva "predictors" o "target"
        predictors = data_select[predictors_col]
        target = data_select[target_col]
        
        x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size=0.25, random_state=13)
        
        tree = DecisionTreeClassifier().fit(x_train, y_train);
        
        print(plot_tree(tree))
        plot.show()
        
        #Evaluar predicciones tomando el porcentaje restante
        predicciones = tree.predict(x_test)
        
        crosstab = pd.crosstab(np.array([y[0] for y in y_test.values.tolist()]), predicciones, rownames = ["Actual"], colnames = ["Predicciones"])
        print(crosstab)
        
        #Porcentaje de precision comparada a informacion real
        print(accuracy_score(y_test, predicciones))
        
print('///////////////////////////////////case 01///////////////////////////////////')    
LinearProgresionData().start();
print('\n///////////////////////////////////case 02///////////////////////////////////\n')
LogisticProgresionData().start();
print('\n///////////////////////////////////case 03///////////////////////////////////\n')
TreeDesicionData().start();
