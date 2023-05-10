ham_words = 0
ham_count = 0
spam_count = 0
spam_words = 0
spam_endswith_exclamation  = 0
fhand = open ('SMSSpamCollection.txt')
for line in fhand :
    line = line.rstrip()
    words = line.strip()
    if line.startswith('ham'):
        ham_words = ham_words + len(words) - 1
        ham_count = ham_count + 1
    else:
        spam_words = spam_words + len(words) - 1
        spam_count = spam_count + 1
        if line.endswith('!'):
            spam_endswith_exclamation = spam_endswith_exclamation + 1
fhand.close ()

print('spam  words: ', round(spam_words/spam_count, 2))
print('ham  words: ', round(ham_words/ham_count, 2))
print( 'spam messages ends with ! ', spam_endswith_exclamation)
