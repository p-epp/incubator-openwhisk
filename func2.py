# -*- coding: utf-8 -*-

# This is Testaction 2

def main(args):
    number = args.get("number")
    square = number+number
    print(square)
    return {"number":square}

#if __name__=="__main__":
#    main({"number":4})