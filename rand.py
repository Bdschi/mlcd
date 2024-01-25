# create a random number between 0  and 10

import random

sum=0
for i in range(1,10):
    for j in range(1,10):
        number = random.randint(0,10)
        sum+=number
        print(number,end=",")
    print(sum)