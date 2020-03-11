# -*- coding: utf-8 -*-
import random
# This is Testaction 3

def main(args):
    square = args.get("number")
    randStr = ""
    for x in range(square):
        randStr += chr(random.randint(32,125))
    print(randStr)
    return {"string":randStr}

#if __name__=="__main__":
#    main({"square":16})