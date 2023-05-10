import math as mth
import statistics


numbers = []
while True:
    print('Enter a number:')
    x=input()
    
    if x == "Done":
        break

    try:
        x = float(x)
    except ValueError:
        print('Not a number')
    

    numbers.append(x)

length = len(numbers)
print(length)

#ispiši: min, max, mean()
print(max(numbers))
print(min(numbers))
print(statistics.mean(numbers))


