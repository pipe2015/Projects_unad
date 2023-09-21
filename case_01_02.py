import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression, LogisticRegression 
import seaborn as sns
import numpy as npy
import pandas as pd

#raw load github
dataDefaulUrl = 'https://raw.githubusercontent.com/pipe2015/Projects_unad/main/';

class loadDataGraphics:
    def __init__(self):
        self.select_data = None;
        self.loadIndex = -1;
        self.list_csv = {
            'regresion_lineal_data': [
                dataDefaulUrl + 'cars.csv' 
            ],
            'regresion_logistica_data': [
                
            ]
        };
    
    def get_data_csv(self, type = "regresion_lineal_data", index = 0, sep = ","): # default parameter type => '', index => 
        self.loadIndex = index;
        try:
            if type in self.list_csv:
                print(self.list_csv.get(type)[index]);
                self.select_data = pd.read_csv(self.list_csv.get(type)[index], sep=sep);
                return self.select_data;
            raise Exception('list csv', 'Not found')
        except Exception as inst:
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
        
    def start(self):
        data_select = self.get_data_csv(self.select_model, 0); 
        
        # print Data view 
        self.viewData();
        
        # view Data Scatter 
        self.loadGraphicsScatter(True, x="year", y='priceUSD');
        
        #Agrego los datos en un array
        years_x = data_select['year'].values.reshape((-1, 1));
        price_y = data_select['priceUSD'];
        #add create model
        model = LinearRegression().fit(years_x, price_y);
        
        print('interseccion (b)', model.intercept_)

        print('Pendiente (m)', model.coef_)
        
        input_list = [[1980], [1990], [2000], [2010]]
        predicciones = model.predict(input_list)
        print(predicciones)
        
        # view load Data Scatter 
        self.loadGraphicsScatter(False, x="year", y="priceUSD", label='Datos originales');
        plot.scatter(input_list, predicciones, color='red')
        plot.plot(input_list, predicciones, color='black', label='Línea de regresión')
        plot.xlabel('year')
        plot.ylabel('priceUSD')
        plot.legend()
        plot.show()
    
    
class LogisticProgresionData(loadDataGraphics): 
    #init 
    def __init__(self):
        super().__init__()
        self.select_model = 'regresion_logistica_data';
    
    def start(self): 
        data_select = self.get_data_csv(self.select_model, 0);
        
        self.getRows();
        
        pass
    
LinearProgresionData().start();