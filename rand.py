# create a random number between 0  and 10

import random
import mlcdconfig

for i in range(1,mlcdconfig.rows1):
    sum=0
    for j in range(1,mlcdconfig.columns):
        number = random.randint(0,10)
        if number>=0:
            sum+=number
            print(number,end=",")
        else:
            sum+=-number
            print("",end=",")
    print(sum)

for i in range(1,mlcdconfig.rows2):
    for j in range(1,mlcdconfig.columns):
        number = random.randint(0,10)
        if number>=0:
            sum+=number
            print(number,end=",")
        else:
            sum+=-number
            print("",end=",")
    print()
