import pandas as pd
import matplotlib.pyplot as plt


#Koliko mjerenja sadrži DataFrame? Kojeg je tipa svaka velicina? Postoje li izostale ili ˇ
#duplicirane vrijednosti? Obrišite ih ako postoje. Kategoricke veli ˇ cine konvertirajte u tip ˇ
#category



data = pd.read_csv('./LV3/data_C02_emission.csv')# UČITAVANJE CSV FILE-A!
lenght = len(data)
print("količina mjerenja:", lenght)#da vidimo koliko imamo podataka


tip = data.info()#printa cijelu tablicu, naziv kolona, broj mjerenja, non-value, itd.
print('tipovi:', tip)


izostale_vrijednosti = data.isnull().sum()#isnull().sum() kaže nam koliko ukupno ima podataka koji su null unutar jedne kolone
print('non-values:', izostale_vrijednosti)


duplikati = data.duplicated().sum()#zbraja koliko je duplikata
print('duplikati:', duplikati)


data.drop_duplicates()#mice duplikate
print('bez duplikata', data)


data.dropna(axis=0)#kada je 0 briše REDOVE, a kad je 1 briše KOLONE u kojima su NULVRIJEDNOSTI


data=data.reset_index()#da nemamo rupe u indexima


for col in data:
    if type(col) == object:
        data[col] = data[col].astype("Category")






#Koja tri automobila ima najvecu odnosno najmanju gradsku potrošnju? Ispišite u terminal: ´
#ime proizvoda¯ ca, model vozila i kolika je gradska potrošnja.


#napravili smo tablicu i u nju upisali ove 3 kolone iz data, SORTITALI prema fuel, od dolje jer je FALSE
tablica = pd.DataFrame(data, columns=['Make', 'Model', 'Fuel Consumption City (L/100km)']).sort_values('Fuel Consumption City (L/100km)', ascending=False)
print(tablica.head(3)) #jer trazimo samo prva i zadnja 3
print(tablica.tail(3))






#Koliko vozila ima velicinu motora izme ˇ du 2.5 i 3.5 L? Kolika je prosje ¯ cna C02 emisija ˇ
#plinova za ova vozila?



print(data[(data['Engine Size (L)'] >= 2.5 ) & (data['Engine Size (L)'] <= 3.5 )].Model.count())#izvlacimo samo KOLONU Model i zbrajamo koliko ih je
new_data = data[(data['Engine Size (L)'] >= 2.5 ) & (data['Engine Size (L)'] <= 3.5 )]['CO2 Emissions (g/km)'].mean()
print(new_data)



#Koliko mjerenja se odnosi na vozila proizvoda¯ ca Audi? Kolika je prosje ˇ cna emisija C02 ˇ
#plinova automobila proizvoda¯ ca Audi koji imaju 4 cilindara?

#25-63
count_audi = data[(data['Make'] == 'Audi')]
print('Broj Audia:', len(count_audi))

emisija = data[(data['Cylinders'] == 4) & (data['Make']=='Audi')]['CO2 Emissions (g/km)'].mean()
print(emisija)






# Koliko je vozila s 4,6,8. . . cilindara? Kolika je prosjecna emisija C02 plinova s obzirom na ˇ
# broj cilindara?

parni_cilindri=data[(data['Cylinders'])%2==0]
print(len(parni_cilindri))
print(parni_cilindri.groupby(by='Cylinders')['CO2 Emissions (g/km)'].mean())






#Kolika je prosjecna gradska potrošnja u slu ˇ caju vozila koja koriste dizel, a kolika za vozila ˇ
# koja koriste regularni benzin? Koliko iznose medijalne vrijednosti?

dizel=data[(data['Fuel Type'] == 'D')]
benzin = data[(data['Fuel Type']=='X') | (data['Fuel Type']=='Z')] 
print(dizel['Fuel Consumption City (L/100km)'].mean())
print('Median', dizel['Fuel Consumption City (L/100km)'].median())
print(benzin['Fuel Consumption City (L/100km)'].mean())
print('Median', benzin['Fuel Consumption City (L/100km)'].median())






#Koje vozilo s 4 cilindra koje koristi dizelski motor ima najvecu gradsku potrošnju goriva?

new=data[(data['Cylinders']==4)&(data['Fuel Type']=='D')].sort_values('Fuel Consumption City (L/100km)', ascending=False)
print(new.head(1))


#Koliko ima vozila ima rucni tip mjenja ˇ ca (bez obzira na broj brzina)?

print('Vozila', len(data[(data['Transmission'].str.startswith('M'))]))


#Izracunajte korelaciju izme ˇ du numeri ¯ ckih veli ˇ cina. Komentirajte dobiveni rezultat
print('KORELACIJA')
print(data.corr(numeric_only=True))





