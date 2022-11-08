#!/usr/bin/env python

import sys

import json

import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell, new_output

import papermill

from huygens.argv import argv


def save_nb(nb, name):
    f = open(name, 'w')
    json.dump(nb, f, indent=1)
    f.close()


def main():

    nb = new_notebook()
    append = lambda s : nb.cells.append(new_code_cell(s))

    append("a=1")
    append("a+5")

    append("from huygens.namespace import *")
    append("from huygens.wiggle import Cell0, Cell1, Cell2")
    append("pink = color.rgba(1.0, 0.37, 0.36, 0.5)")
    append("Cell0('n', fill=pink)")

    save_nb(nb, "test.ipynb")
    nb = papermill.execute_notebook("test.ipynb", None, kernel_name="python3")
    save_nb(nb, "test_exec_1.ipynb")

    if 0:
        text, image = ['text/plain', 'image/svg+xml']
        #print(nb)
        for cell in nb.cells:
            for output in cell["outputs"]:
                data = output["data"]
                #print(list(data.keys()))
                if image in data and text in data:
                    item = data[text]
                    del data[text]
                    data[text] = [item]
                if image in data:
                    item = data[image]
                    item = [item+"\n" for item in item.split("\n")]
                    data[image] = item



def convert_to_py(name, output=None):

    assert name.endswith(".ipynb")

    s = open(name).read()
    nb = nbformat.reads(s, as_version=4)

    output = sys.stdout if output is None else open(output, 'w')

    for cell in nb.cells:
        #print(list(cell.keys()))
        source = cell["source"]
        tp = cell["cell_type"]
        if tp == "markdown":
            lines = source.split('\n')
            source = '\n'.join("# "+line for line in lines)
        print(source+"\n", file=output)

def convert_to_nb(name, output):

    assert name.endswith(".py")
    assert output and output.endswith(".ipynb")

    s = open(name).read()
    lines = s.split('\n')

    lines.append("# ") # trigger last code block

    nb = new_notebook()
    add_code = lambda s : nb.cells.append(new_code_cell(s))
    add_md = lambda s : nb.cells.append(new_markdown_cell(s))

    code = None
    md = None
    for line in lines:
        if line.startswith("# ") and md is None:
            if code is not None:
                while code and not code[-1].strip():
                    code.pop()
                if code:
                    code = '\n'.join(code)
                    add_code(code)
                code = None
            md = []
        if line.startswith("# "):
            assert code is None
            line = line[2:]
            md.append(line)
        else:
            if code is None:
                if md is not None:
                    md = '\n'.join(md)
                    if md.strip():
                        add_md(md)
                    md = None
                code = []
            if code or line.strip():
                code.append(line)


    save_nb(nb, output)
    nb = papermill.execute_notebook(output, None, kernel_name="python3")
    save_nb(nb, output)



if __name__ == "__main__":

    if argv.convert_to_py:
        convert_to_py(argv.next(), argv.next())

    elif argv.convert_to_nb:
        convert_to_nb(argv.next(), argv.next())


