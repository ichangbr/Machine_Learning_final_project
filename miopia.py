import pandas as pd
import numpy as np
import pickle

class Model:
    def __init__(self,x_path):
        self.data = pd.read_csv(x_path)
        self.model = pickle.load(open("svm_linear_model.pickle", 'rb'))
        self.clean_data()

    def del_columns1(self):
        #Delete columns that are not in the model
        to_del = ['origen antepasados (extranjeros)', 'hº act.cerca sem','fototipo',
                    'Grupo fot','fecha','origen antepasados (españa)']
        for col in to_del:
            del self.data[col]
    
    def fill_NAs(self):
        to_fill = ['hº ocio exteriores sem', 'horas interior sem','Familiar miope num.','familiar miope','pat. Ret. Miop magna']
        modes = [0,60,1,'No lo se','No lo se']
        for col,moda in zip(to_fill,modes):
            self.data[col].fillna(moda,inplace=True)
    
    def create_dummies(self):
        #Create and clean dummies
        binary_columns = []
        for col in self.data.select_dtypes(include = [object]):
            if len(self.data[col].unique()) == 2:
                binary_columns.append(col)
        X_prov = self.data.loc[:,~self.data.columns.isin(binary_columns)].copy()
        dummies = pd.get_dummies(X_prov)
        binary_data = self.data.loc[:,binary_columns]
        for col in binary_columns[1:]:
          binary_data[col] = (binary_data[col] == 'SI').astype(int).astype(object)

        binary_data['sexo'] = (binary_data['sexo'] == 'Mujer').astype(int).astype(object) # Mujer = 1, Hombre = 0
        dummies = pd.concat([dummies,binary_data], axis = 1)
        dummies.rename(columns = {'sexo':'Mujer'}, inplace = True)

        #Clean dummies with redundant columns
        #familiar miope
        import re #for regex operations
        familiar_miope_names = list(dummies.columns)
        reg = re.compile(r'familiar miope_.*')
        familiar_miope_names = list(filter(reg.search,familiar_miope_names))
        reg = re.compile(r'.*,.*')
        to_change = list(filter(reg.search,familiar_miope_names))
        for change in to_change:
            other_columns = change.split("_")[1].split(", ")
            other_columns = [f'familiar miope_{col}' for col in other_columns]
            select_list = dummies[change].ne(0)
            dummies.loc[select_list,other_columns] = 1
        dummies.drop(to_change,axis = 1, inplace = True)
        
        #familiar miope magno
        select_list = dummies['familiar miope magno_No lo se, No'].ne(0)
        dummies.loc[select_list,'familiar miope magno_No lo se'] = 1
        dummies.drop('familiar miope magno_No lo se, No',axis = 1, inplace = True)
        familiar_miope_names = list(dummies.columns)
        reg = re.compile(r'familiar miope magno_.*')
        familiar_miope_names = list(filter(reg.search,familiar_miope_names))
        reg = re.compile(r'.*,.*')
        to_change = list(filter(reg.search,familiar_miope_names))
        
        for change in to_change:
            other_columns = change.split("_")[1].split(", ")
            other_columns = [f'familiar miope_{col}' for col in other_columns]
            select_list = dummies[change].ne(0)
            dummies.loc[select_list,other_columns] = 1
        dummies.drop(to_change,axis = 1, inplace = True)
        return dummies

    def separate_dummies(self):
        self.numeric = self.dummies_df.select_dtypes(include=[np.float64, np.int64])
        self.categorical = self.dummies_df.select_dtypes(exclude=[np.float64, np.int64])

    def scale_numeric(self):
        with open('min_max.pickle','rb') as f:
            min_max_data = pickle.load(f)
        scaled = {}
        for col in min_max_data:
            scaled[col] = (self.numeric[col] - min_max_data[col][1])/(min_max_data[col][0]-min_max_data[col][1])
        self.numeric = pd.DataFrame(scaled)


    def join_final(self):
        self.final = pd.concat([self.numeric,self.categorical], axis=1)
        keep = ['hº deporte sem', 'horas exterior sem', 'Familiar miope num.', 'LA Media', 'Querato OD', 'OD-Media',
                'OS-Media', 'OS-DS', 'CUVAF-Media', 'CUVAF-DS', 'Promedio Nasal', 'Promedio Temporal',
                'familiar miope_Hermanos', 'familiar miope_Madre', 'familiar miope_No', 'familiar miope_Padre',
                'Familiar MM SI/NO_No', 'Familiar MM SI/NO_No lo se', 'Familiar MM SI/NO_Si',
                'familiar miope magno_Hermanos', 'familiar miope magno_Madre', 'familiar miope magno_No',
                'familiar miope magno_No lo se', 'pat. Ret. Miop magna_No', 'pat. Ret. Miop magna_No lo se',
                'pat. Ret. Miop magna_Sí, en ambos ojos','toma sol']
        self.final = self.final[keep]
    
    
    def clean_data(self):
        self.del_columns1()
        self.fill_NAs()
        self.dummies_df = self.create_dummies()
        self.separate_dummies()
        self.scale_numeric()
        self.join_final()
    
    def predict(self):
        prediction = self.model.predict(self.final)
        translate_dict = {
            'MM': ['SI','SI','MM','MM'],
            'M1': ['SI','NO','M','M1'],
            'M2': ['SI','NO','M','M2'],
            'C': ['NO','NO','C','C']
        }

        results_dict = {
            'M':[translate_dict[tag][0] for tag in prediction],
            'MM':[translate_dict[tag][1] for tag in prediction],
            'Combo':[translate_dict[tag][2] for tag in prediction],
            'DCombo':[translate_dict[tag][3] for tag in prediction]
        }

        self.results = pd.DataFrame(results_dict)
        print(self.results)

    def save_prediction(self):
        self.results.to_csv('prediction.csv', index=False)




