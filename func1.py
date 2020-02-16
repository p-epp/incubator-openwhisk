# -*- coding: utf-8 -*-
import time
# This is Testaction 1

def main(args):
    string = args.get("string")
    number = len(string)
    print(number)
    print("sleep")
    time.sleep(10)
    return {"number": number, "sleep": "sleep"}

#if __name__=="__main__":
#    main({"string":"blub"})


