import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('./LV3/data_C02_emission (1).csv')


#Pomocu histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz.

plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist')
plt.show ()






#Pomocu dijagrama raspršenja prikažite odnos izme ´ du gradske potrošnje goriva i emisije ¯
#C02 plinova. Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose izmedu¯
#velicina, obojite to ˇ ckice na dijagramu raspršenja s obzirom na tip goriva

data['Fuel Color'] = data['Fuel Type'].map(
   {
    "X" : "Red",
    "Z" : "Blue",
    "D" : "Green",
    "E" : "Purple",
    "N" : "Yellow"
   }
)
data.plot.scatter(x='Fuel Consumption City (L/100km)', y='CO2 Emissions (g/km)', c='Fuel Color')
plt.show()






#Pomocu kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip ´
#goriva. Primjecujete li grubu mjernu pogrešku u podacima?

data.boxplot(column=['Fuel Consumption Hwy (L/100km)'], by='Fuel Type')
plt.show()






#Pomocu stup ´ castog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu ˇ
#groupby.

new_data = data.groupby("Fuel Type")["Cylinders"].count().plot(kind="bar")
plt.show()






#Pomocu stup ´ castog grafa prikažite na istoj slici prosje ˇ cnu C02 emisiju vozila s obzirom na ˇ
#broj cilindara.

data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean().plot(kind="bar", ax=new_data)
plt.show()