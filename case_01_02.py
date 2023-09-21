import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression, LogisticRegression 
import seaborn as sns
import numpy as npy
import pandas as pd
import numpy as np

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
    
    def start(self): 
        data_select = self.get_data_csv(self.select_model, 0);
        
        self.getRows();
        
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

print('///////////////////////////////////case 01///////////////////////////////////')    
LinearProgresionData().start();
print('\n///////////////////////////////////case 02///////////////////////////////////\n')
LogisticProgresionData().start();