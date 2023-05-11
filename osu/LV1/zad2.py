import math as mth

x = 0.0
print('Upišite ocjenu od 0.0-1.0:')
while True:
    x=float(input())
    try:
        x=float(x)
    except ValueError:
        print('Number not in range')
    if 0.0<=x<=1.0:
        break
    else:
        print('Enter number in valid range')

if x>=0.9:
    print('A')
elif x>=0.8:
    print('B')
elif x>=0.7:
    print('C')
elif x>=0.6:
    print('D')
elif x<0.6:
    print('F')

    



