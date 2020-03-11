# -*- coding: utf-8 -*-
import time
import random
# This is Testaction 1

def main(args):
    funcCount = int(args.get("count")) + 1
    mode = str(args.get("mode"))
    mean = float(args.get("mean"))
    newMean = mean
    if mode != "homogenous":
        newMean = 0.5 - 0.05*((funcCount-1)/2) if funcCount % 2 == 1 else 0.5 + 0.05*(funcCount/2)
    randomValue = random.normalvariate(newMean, 0.05)
    print(randomValue)
    time.sleep(randomValue)
    return {"count": funcCount, "mean": mean, "mode": mode}

#main(main(main()))

