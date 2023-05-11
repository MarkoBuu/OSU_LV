import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score


data = pd.read_csv('./LV4/data_C02_emission.csv')


#Odaberite željene numericke veli ˇ cine speci ˇ ficiranjem liste s nazivima stupaca. Podijelite podatke 
#na skup za ucenje i skup za testiranje u omjeru 80%-20%.

input_variables = [                                     #na X jer su to podatci prema kojima računamo
    'Fuel Consumption City (L/100km)',
    'Fuel Consumption Hwy (L/100km)',
    'Fuel Consumption Comb (L/100km)',
    'Fuel Consumption Comb (mpg)',
    'Engine Size (L)',
    'Cylinders']

output_variable = ['CO2 Emissions (g/km)']              #na Y tj podatak koji predviđamo

X = data[input_variables].to_numpy()
y = data[output_variable].to_numpy()
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2 , random_state =1 )

plt.scatter(x=X_train[:, 0], y=y_train, c="b")          #ne ukljucuje krajnji element, ali pocinje od 0, da je pisalo 3:5 uzme od 3-4
plt.scatter(x=X_test[:, 0], y=y_test, c="r")
plt.show()






#Pomocu matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova ´
#o jednoj numerickoj veli ˇ cini. Pri tome podatke koji pripadaju skupu za u ˇ cenje ozna ˇ cite

sc = MinMaxScaler()                                             #SKALIRANJE VELICINA
X_train_n = sc.fit_transform( X_train )                         #namjesti veličine i skalira da bi bilo izmedu 0 i 1 
X_test_n = sc.transform( X_test )                               #transformira veličine tj. skalira i testne podatke kako bi bili kompatibil i sa onima iz train skupa

for i in range(len(input_variables)):
    
    ax1 = plt.subplot(211)                                      #2 reda, 1 stupac i 1 index
    ax1.hist(x=X_train[:, i])                   
    ax2 = plt.subplot(212)                                      #2 reda, 1 stupca i 2 index
    ax2.hist(x=X_train_n[:, i])
    plt.show()






#c) i d)

linearModel = lm.LinearRegression()
linearModel.fit( X_train_n , y_train )                          #TRENIRANJE MODELA!!!!!
print(linearModel.coef_)                                        #koeficjenti uz x
print(linearModel.intercept_)                                   #koeficjent nulti hvata (nulti parametar)





#e)
y_test_p = linearModel.predict( X_test_n )                      #predviđanje testnog skupa
plt.scatter(x=X_train_n[:, 0], y=y_train, c="b")                
plt.scatter(x=X_test_n[:, 0], y=y_test, c="r")
plt.show()






#f)
MAE = mean_absolute_error(y_test, y_test_p)                     #ERRORI!!!!!!!!!!!!!!!!
RMSE = mean_squared_error(y_test, y_test_p, squared=False)      #VRSTE FUNKCIJA ZA IZRAČUN ERORA
MAPE = mean_absolute_percentage_error(y_test, y_test_p)
R2 = r2_score(y_test, y_test_p)

plt.scatter(x=X_train_n[:, 0], y=y_train, c="b")
plt.scatter(x=X_test_n[:, 0], y=y_test, c="r")
plt.show()

print("MAE", MAE)
print("RMSE", RMSE)
print("MAPE", MAPE)
print("R2", R2)


