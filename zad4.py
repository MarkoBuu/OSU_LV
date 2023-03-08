file = open('song.txt')
dictin = dict()


for line in file:
    line = line.rstrip()
    words = line.split()
    for word in words:
        if word not in dictin:
            dictin[word] = 1
        else:
            dictin[word] += 1

counter = 0
for key, val in dictin.items():
    if val == 1:
        counter += 1
        print(key)
print(dictin)
print(counter)
file.close()