# -*- coding: utf-8 -*-
import time
# This is Testaction 1

def main(args):
    string = args.get("string")
    number = len(string)
    time.sleep(15)
    return {"number": number}

