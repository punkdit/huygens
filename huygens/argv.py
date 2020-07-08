
import sys

class Argv(object):
    def __init__(self):
        self.args = sys.argv[:]
        self._args = []
        self.kw = self.argmap = {}
        for arg in sys.argv:
            if '=' in arg:
                items = arg.split('=')
                name, val = items[0], '='.join(items[1:])
                self.argmap[name] = self.parse(val)
                self.args.remove(arg)

    def next(self):
        return self.args.pop(1) if len(self.args)>1 else None

    def parse(self, value):
        try:
            return eval(value)
        except:
            return value

    def get(self, name, default=None):
        return self.argmap.get(name, default)

    def __str__(self):
        return ' '.join(sys.argv)

    def __getattr__(self, name):
        value = self.argmap.get(name)
        if value is None:
            if name in self.args:
                self.args.remove(name)
                value = True
                setattr(self, name, True)
        return value

    def __len__(self):
        return len(self.args)

    def __getitem__(self, index):
        return self.args[index]

    def __len__(self):
        return len(self.args)


argv = Argv()

