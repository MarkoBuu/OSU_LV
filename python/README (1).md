# Osnove-strojnog-u-enja---lv

1. Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo
1912. godine. Upoznajte se s datasetom i dodajte programski kod u skriptu pomo´cu kojeg možete
odgovoriti na sljede´ca pitanja:
      
      a) Za koliko osoba postoje podatci u ovom skupu podataka?
      b) Koliko je osoba preživjelo potonu´ce broda?
      c) Pomo´cu stupˇcastog dijagrama prikažite postotke preživjelih muškaraca i žena. Dodajte
          nazive osi i naziv dijagrama. Komentirajte korelaciju spola i postotka preživljavanja.
      d) Kolika je prosjeˇcna dob svih preživjelih žena, a kolika je prosjeˇcna dob svih preživjelih
          muškaraca?
      e) Koliko godina ima najmlad¯i preživjeli muškarac u svakoj od klasa? Komentirajte.

2.Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo
1912. godine. Upoznajte se s datasetom. Uˇcitajte dane podatke. Podijelite ih na ulazne podatke X
predstavljene stupcima Pclass, Sex, Fare i Embarked i izlazne podatke y predstavljene stupcem
Survived. Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 60:40.
Izbacite izostale i null vrijednosti. Skalirajte podatke. Dodajte programski kod u skriptu pomo´cu
kojeg možete odgovoriti na sljede´ca pitanja:

      a) Izradite algoritam KNN na skupu podataka za uˇcenje (uz K=5). Vizualizirajte podatkovne
          primjere i granicu odluke.
      b) Izraˇcunajte toˇcnost klasifikacije na skupu podataka za uˇcenje i skupu podataka za testiranje.
          Komentirajte dobivene rezultate.
      c) Pomo´cu unakrsne validacije odredite optimalnu vrijednost hiperparametra K algoritma
          KNN.
      d) Izraˇcunajte toˇcnost klasifikacije na skupu podataka za uˇcenje i skupu podataka za testiranje
          za dobiveni K. Usporedite dobivene rezultate s rezultatima kada je K=5.
          
3.Datoteka titanic.csv sadrži podatke o putnicima broda Titanic, koji je potonuo
1912. godine. Upoznajte se s datasetom. Uˇcitajte dane podatke. Podijelite ih na ulazne podatke X
predstavljene stupcima Pclass, Sex, Fare i Embarked i izlazne podatke y predstavljene stupcem
Survived. Podijelite podatke na skup za uˇcenje i skup za testiranje modela u omjeru 80:20.
Izbacite izostale i null vrijednosti. Skalirajte podatke. Dodajte programski kod u skriptu pomo´cu
kojeg možete odgovoriti na sljede´ca pitanja:

      a) Izgradite neuronsku mrežu sa sljede´cim karakteristikama:
        - model oˇcekuje ulazne podatke X
        - prvi skriveni sloj ima 16 neurona i koristi relu aktivacijsku funkciju
        - drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju
        - tre´ci skriveni sloj ima 4 neurona i koristi relu aktivacijsku funkciju
        - izlazni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
          Ispišite informacije o mreži u terminal.
      b) Podesite proces treniranja mreže sa sljede´cim parametrima:
          - loss argument: binary_crossentropy
          - optimizer: adam
          - metrika: accuracy.
      c) Pokrenite uˇcenje mreže sa proizvoljnim brojem epoha (pokušajte sa 100) i veliˇcinom
          batch-a 5.
      d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju uˇcitanog modela.
      e) Izvršite evaluaciju mreže na testnom skupu podataka.
      f) Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup
          podataka za testiranje. Komentirajte dobivene rezultate i predložite kako biste ih poboljšali,
          ako je potrebno.
