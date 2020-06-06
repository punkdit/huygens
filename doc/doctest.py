#!/usr/bin/env python3

import sys
import ast

from meta.asttools import print_ast


def main(name):

    source = open(name).read()
    node = ast.parse(source, name)

    print(node)
    s = ast.dump(node)
    print(node.body)

    print_ast(node)
    


if __name__ == "__main__":

    name = sys.argv[1]
    main(name)


