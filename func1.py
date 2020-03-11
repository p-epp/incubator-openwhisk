# -*- coding: utf-8 -*-
import time
import numpy as np
# This is Testaction 1

def main(args):
    string = args.get("string")
    number = len(string)
    randomValue = np.random.normal(0.5, 0.05)
    #print("Here Comes Something {}".format(randomValue))
    time.sleep(np.random.normal(0.5, 0.05))
    return {"number": number}

#main({"string":"phillip"})

