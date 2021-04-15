import os
import torch

import model

def main():
    print("Hello World!")
    net = model.Net(4, [64, 64, 64], 4, dropout_p=0.5)
    x = net.forward([1,2,3,4])
    print(x)

if __name__ == "__main__":
    main()