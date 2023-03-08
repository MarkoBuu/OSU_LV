list = []

while True:
    try : 
        br = input("Unesite broj: ")
        if br == "Done":
            break
        list.append(float(br))
    except:
        print("Unesite valjani broj")


list.sort()
print(list)

print("Broj brojeva je: " + str(len(list)))

print("Najveca vrijednost u listi: "+ str(max(list)) )

print("Najmanja vrijednost u listi: "+ str(min(list)) )

n = sum(list) / len(list)
print("Srednja vrijednost brojeva u list je: " + str(n))