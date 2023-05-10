
fhand = open('song.txt')
words= []
for line in fhand :
    line = line.rstrip()
    for word in line.split():
        words.append(word)
    
word_set = set(words)
dictionary = {}

for word in word_set:
    dictionary[word]=0

for word in words:
    dictionary[word] = dictionary[word]+1


print(dictionary)

fhand.close()