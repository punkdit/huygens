#!/usr/bin/env python3

import os
import collections

import bruhat.render.doc
from bruhat.render.front import Canvas, Scale
from bruhat.render.boxs import Box


class TestRun(object):
    def __init__(self, func, start=None, end=None, img=None, result=None):
        self.func = func
        self.start = start
        self.end = end
        self.img = img
        self.result = result


all_names = set()
counter = 0


def run_test(func, dummy=False):

    global counter

    items = func()

    if not isinstance(items, collections.Iterator):
        yield TestRun(func, func.__code__.co_firstlineno, result=items)
        return

    start = items.gi_frame.f_lineno # index

    while 1:
      try:
        box = None
        cvs = None
        name = None

        result = items.__next__()

        if isinstance(result, tuple):
            box, name = result
        elif isinstance(result, Box):
            box = result
        elif isinstance(result, Canvas):
            cvs = result
        else:
            assert 0, "%r not understood" % (result,)

        if not name:
            name = "output-%d"%counter
            counter += 1
        
        assert name not in all_names, "name dup: %r"%name
        all_names.add(name)

        svgname = "images/%s.svg"%name
        pdfname = "images/%s.pdf"%name

        end = items.gi_frame.f_lineno-1 # index
        test = TestRun(func, start, end, svgname)
        yield test
        start = end+1

        if dummy:
            continue

        try:
            print("run_tests: rendering", name, func)
            if cvs is None:
                cvs = Canvas()
                cvs.append(Scale(2.0))
                box.render(cvs)
            else:
                cvs = Canvas([Scale(2.0), cvs])
            
            cvs.writeSVGfile(svgname)
            cvs.writePDFfile(pdfname)
            print()
        except:
            print("run_tests: render failed for", name, func)
            raise

      except StopIteration:
        break



def harvest(path, name, dummy=False):
    print("run_tests.harvest", name)
    assert name.endswith(".py")
    stem = name[:-len(".py")]
    desc = "bruhat.render.doc."+stem
    __import__(desc)
    m = getattr(bruhat.render.doc, stem)
    funcs = []
    for attr in dir(m):
        value = getattr(m, attr)
        if attr.startswith("test_") and isinstance(value, collections.Callable):
            funcs.append(value)

    funcs.sort(key = lambda f : (f.__module__, f.__code__.co_firstlineno))
    for func in funcs:
        for test in run_test(func, dummy=dummy):
            yield test


def run():

    path = os.path.dirname(__file__)
    names = os.listdir(path)
    names = [name for name in names 
        if name.endswith(".py") and name.startswith("test_")]
    names.sort()

    for name in names:
        for test in harvest(path, name):
            yield test


def main():
    for test in run():
        pass

    print("run_tests.main: finished")


if __name__ == "__main__":
    main()


