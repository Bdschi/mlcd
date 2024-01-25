# create a random number between 0  and 10

import random

rows1=100
rows2=100
columns=10
for i in range(1,rows1):
    sum=0
    for j in range(1,columns):
        number = random.randint(-10,10)
        if number>=0:
            sum+=number
            print(number,end=",")
        else:
            sum+=-number
            print("",end=",")
    print(sum)

for i in range(1,rows2):
    for j in range(1,columns):
        number = random.randint(-10,10)
        if number>=0:
            sum+=number
            print(number,end=",")
        else:
            sum+=-number
            print("",end=",")
    print()
