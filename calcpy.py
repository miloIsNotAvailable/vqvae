import sys

class Parse:
    def __init__( self ):
        self.a=int(sys.argv[1])
        self.b=int(sys.argv[3])
        self.sign=sys.argv[2]

    def __call__( self ):
        if self.sign == "+":
                return self.a + self.b
        elif self.sign == "-":
                return self.a - self.b
        elif self.sign == "*":
                return self.a * self.b
        elif self.sign == "/":
                return self.a / self.b
        else:
            print( "?" ) 
                

p = Parse()
print(p())
