# Osnove-strojnog-u-enja---lv

1. Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
otkrivanja dijabetesa, pri cemu se u devetom stupcu nalazi klasa 0 (nema dijabetes) ili klasa 1 ˇ
(ima dijabetes). Ucitajte dane podatke u obliku numpy polja ˇ data. Dodajte programski kod u
skriptu pomocu kojeg možete odgovoriti na sljede ´ ca pitanja:

      a) Na temelju velicine numpy polja ˇ data, na koliko osoba su izvršena mjerenja
      b) Postoje li izostale ili duplicirane vrijednosti u stupcima s mjerenjima dobi i indeksa tjelesne
          mase (BMI)? Obrišite ih ako postoje. Koliko je sada uzoraka mjerenja preostalo?
      c) Prikažite odnos dobi i indeksa tjelesne mase (BMI) osobe pomocu´ scatter dijagrama.
          Dodajte naziv dijagrama i nazive osi s pripadajucim mjernim jedinicama. Komentirajte ´
          odnos dobi i BMI prikazan dijagramom.
      d) Izracunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost indeksa tjelesne ˇ
          mase (BMI) u ovom podatkovnom skupu.
      e) Ponovite zadatak pod d), ali posebno za osobe kojima je dijagnosticiran dijabetes i za one
          kojima nije. Kolikom je broju ljudi dijagonosticiran dijabetes? Komentirajte dobivene
          vrijednosti.
          
2.Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
otkrivanja dijabetesa, pri cemu se u devetom stupcu nalazi izlazna veli ˇ cina, predstavljena klasom ˇ
0 (nema dijabetes) ili klasom 1 (ima dijabetes).
Ucitajte dane podatke u obliku numpy polja ˇ data. Podijelite ih na ulazne podatke X i izlazne
podatke y. Podijelite podatke na skup za ucenje i skup za testiranje modela u omjeru 80:20. ˇ
Dodajte programski kod u skriptu pomocu kojeg možete odgovoriti na sljedeca pitanja:

    a) Izgradite model logisticke regresije pomo ˇ cu scikit-learn biblioteke na temelju skupa poda- ´
      taka za ucenje. ˇ
    b) Provedite klasifikaciju skupa podataka za testiranje pomocu izgra ´ denog modela logisti ¯ cke ˇ
      regresije.
    c) Izracunajte i prikažite matricu zabune na testnim podacima. Komentirajte dobivene rezul- ˇ
      tate.
    d) Izracunajte to ˇ cnost, preciznost i odziv na skupu podataka za testiranje. Komentirajte ˇ
      dobivene rezultate.
      
3.Datoteka pima-indians-diabetes.csv sadrži mjerenja provedena u svrhu
otkrivanja dijabetesa, pri cemu je prvih 8 stupaca ulazna veli ˇ cina, a u devetom stupcu se nalazi ˇ
izlazna velicina: klasa 0 (nema dijabetes) ili klasa 1 (ima dijabetes). ˇ
Ucitajte dane podatke. Podijelite ih na ulazne podatke ˇ X i izlazne podatke y. Podijelite podatke
na skup za ucenje i skup za testiranje modela u omjeru 80:20.

    a) Izgradite neuronsku mrežu sa sljedecim karakteristikama: ´
      - model ocekuje ulazne podatke s 8 varijabli ˇ
      - prvi skriveni sloj ima 12 neurona i koristi relu aktivacijsku funkciju
      - drugi skriveni sloj ima 8 neurona i koristi relu aktivacijsku funkciju
      - izlasni sloj ima jedan neuron i koristi sigmoid aktivacijsku funkciju.
        Ispišite informacije o mreži u terminal.
    b) Podesite proces treniranja mreže sa sljedecim parametrima: ´
      - loss argument: cross entropy
       - optimizer: adam
        - metrika: accuracy.
    c) Pokrenite ucenje mreže sa proizvoljnim brojem epoha (pokušajte sa 150) i veli ˇ cinom ˇ
        batch-a 10.
    d) Pohranite model na tvrdi disk te preostale zadatke izvršite na temelju ucitanog modela. ˇ
    e) Izvršite evaluaciju mreže na testnom skupu podataka.
    f) Izvršite predikciju mreže na skupu podataka za testiranje. Prikažite matricu zabune za skup
        podataka za testiranje. Komentirajte dobivene rezultate
