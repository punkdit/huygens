#!/usr/bin/env python3

import os
from random import random
from time import sleep

from huygens import config
config(text="xelatex")

from huygens.base import SCALE_CM_TO_POINT

from huygens.front import *
from huygens.front import _defaultlinewidth
from huygens.box import *
from huygens.turtle import Turtle
from huygens.loadsvg import loadsvg
from huygens.argv import argv

FPS = 25. # default ffmpeg encoding framerate

"""
For 16:9 aspect ratio, encode at these resolutions:

2160p: 3840x2160
1440p: 2560x1440
1080p: 1920x1080
720p: 1280x720
480p: 854x480
360p: 640x360
240p: 426x240
"""

bg = color.rgb.white
bg = color.rgb.grey
#bg = color.rgb(1.,1., 0.)
fg = color.rgb.black

black = color.rgb(0., 0., 0., 1.0)
white = color.rgb(1., 1., 1., 1.0)
blue = color.rgb(0.4, 0.3, 0.9, 1.0)
orange = color.rgb(0.8, 0.2, 0.2, 1.0)

W, H = 854, 480
width, height = W/SCALE_CM_TO_POINT, H/SCALE_CM_TO_POINT
#print(width, height)


def arrow(cvs, x0, y0, x1, y1, size, attrs):
    t = Turtle(x0, y0, attrs=attrs)
    t.moveto(x1, y1)
    t.arrow(size, style="curve")
    t.stroke(cvs=cvs)


def setup():
    cvs = canvas.canvas()

    p = path.path([
        path.moveto(0., 0.),
        path.lineto(0., height),
        path.lineto(width, height),
        path.lineto(width, 0),
        path.closepath()])
    cvs.clip(p)
    cvs.fill(p, [bg])

    #cvs.stroke(path.line(0., 0., width, height), [fg])
    #cvs.stroke(path.line(0., height, width, 0), [fg])

    return cvs


def text_box(s, scale=5., color=black):
    sub = canvas.canvas()
    sub.text(0., 0., s, [color, trafo.scale(scale)])
    box = CanBox(sub)
    box = AlignBox(box, "center")
    return box


def title_seq():

    x0, y0 = 0.5*width, 0.6*height
    x1, y1 = 0.6*width, 0.3*height

    #x0, y0 = 1.5*width, 1.6*height
    #x1, y1 = 1.6*width, 1.3*height

    spread = 1.0

    N = 100
    while 1:

        cvs = canvas.canvas()

        #color = RGBA(r, g, b, a)
        color = black
        text_box("What the Quantum ?!?", color=color).render(cvs, x0, y0)

        text_box("Episode 1").render(cvs, x1, y1)

        x0 += 0.2/N*width
        x1 -= 0.2/N*width
    
        yield cvs

        spread *= 0.9



def axis(cvs, x, y, w, h, attrs):

    attrs = [style.linejoin.round, style.linecap.round] + attrs

    size = 0.7
    arrow(cvs, x-0.1*w, y, x+1.1*w, y, size, attrs)
    arrow(cvs, x, y-0.1*h, x, y+1.1*h, size, attrs)


class Axis(Canvas):
    def __init__(self, x0, y0, dx, dy, lw=0.02):
        Canvas.__init__(self)
        attrs = [fg, style.linewidth.THIck]
        axis(self, x0, y0, dx, dy, attrs)
        self.append(trafo.translate(x0, y0))
        self.append(trafo.scale(dx, dy))
        self.append(LineWidth(lw/dx))


def lattice_balls(r, theta):
    dx, dy = 2*sin(theta), 2*cos(theta)
    balls = []
    for i in range(7): # row
      for j in range(-3, 7): # col
        balls.append(((1. + dx*i + 2.*j)*r, (1. + dy*i)*r))
    return balls


def lin(i, x0, x1, N):
    assert 0<=i<N
    alpha = i / (N-1)
    return (1-alpha)*x0 + alpha*x1


def ilin(x0, x1, N):
    for i in range(N):
        assert 0<=i<N
        alpha = i / (N-1)
        yield (1-alpha)*x0 + alpha*x1


def iconst(x0, N):
    for i in range(N):
        yield x0


class Seq(object):
    def __init__(self, N):
        self.N = N
    def __str__(self):
        return "%s(%s)"%(self.__class__.__name__, self.N)
    __repr__ = __str__
    def __len__(self):
        return self.N
    def __getitem__(self, idx):
        raise IndexError
    @classmethod
    def promote(cls, item):
        if isinstance(item, Seq):
            return item
        if isinstance(item, list):
            return ConcSeq(item)
        raise TypeError
    def __add__(lhs, rhs):
        return AddSeq(lhs, rhs)
    def __mul__(lhs, rhs):
        return MulSeq(lhs, rhs)

class ConcSeq(Seq):
    def __init__(self, items):
        Seq.__init__(self, len(items))
        self.items = items
    def __getitem__(self, idx):
        return self.items[idx]


class PairSeq(Seq):
    def __init__(self, lhs, rhs):
        lhs = Seq.promote(lhs)
        rhs = Seq.promote(rhs)
        Seq.__init__(self, lhs.N + rhs.N)
        self.lhs = lhs
        self.rhs = rhs

class AddSeq(PairSeq):
    def __getitem__(self, idx):
        lhs, rhs = self.lhs, self.rhs
        if 0<=idx<lhs.N:
            return lhs[idx]
        elif lhs.N<=idx<lhs.N+rhs.N:
            return rhs[idx-lhs.N]
        else:
            raise IndexError

class MulSeq(PairSeq):
    def __getitem__(self, idx):
        lhs, rhs = self.lhs, self.rhs
        return (lhs[idx], rhs[idx])


class Lin(Seq):
    def __init__(self, x0, x1, N):
        Seq.__init__(self, N)
        self.x0 = x0
        self.x1 = x1
    def __getitem__(self, idx):
        N = self.N
        if idx<0 or idx>=N:
            raise IndexError
        x0, x1 = self.x0, self.x1
        alpha = idx / (N-1)
        return (1-alpha)*x0 + alpha*x1


class Smooth(Lin):
    def __getitem__(self, idx):
        N = self.N
        if idx<0 or idx>=N:
            raise IndexError
        x0, x1 = self.x0, self.x1
        x = idx / (N-1) # 0.<=x<=1.
        y = -2*(x**3) + 3*(x**2) # 0.<=y<=1.
        return (1-y)*x0 + y*x1


def ball_seq_1():

    N = 100

    r = 1./12

    rs = Smooth(r, 0.5*r, 2*N)
    thetas = Smooth(0., pi/6, N) + Smooth(pi/6, 0., N)
    for (r, theta) in rs*thetas:

        x, y, w = 0.1*width, 0.1*height, 0.7*height
        axis = Axis(x, y, w, w, lw=0.04)

        p = path.rect(0, 0, 1., 1.)
        axis.stroke(p, [linestyle.dashed, orange])
        axis.clip(p)

        balls = lattice_balls(r, theta)
        for (x1, y1) in balls:
            p = path.circle(x1, y1, r)
            axis.fill(p, [blue])
            axis.stroke(p)

        yield axis


class Ball(object):
    def __init__(self, x, y, r, dx, dy, fill):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.ddx = 0.
        self.ddy = 0.
        self.r = r
        self.fill = fill

    def update(self, ddx=0., ddy=0.):
        self.x += self.dx
        self.y += self.dy
        self.dx += self.ddx
        self.dy += self.ddy

    def wrap(self, x0, y0, x1, y1):
        x, y = self.x, self.y
        if x < x0:
            x = x1
        elif x > x1:
            x = x0
        if y < y0:
            y = y1
        elif y > y1:
            y = y0

        self.x, self.y = (x, y)

    def stroke(self, cvs):
        p = path.circle(self.x, self.y, self.r)
        cvs.stroke(p)
        
    def render(self, cvs, fill=None):
        p = path.circle(self.x, self.y, self.r)
        fill = fill or self.fill
        if fill is not None:
            cvs.fill(p, [fill])
        cvs.stroke(p)


def rnd(a, b):
    return (b-a)*random() + a


def ball_seq_2():

#    blue = color.rgb(0.4, 0.3, 0.9, 0.8)
#    orange = color.rgb(0.8, 0.2, 0.2, 0.8)

    radius = 1.5

    N = 20
    balls = []
    def mk_ball():
        x, y = width*random(), height + radius
        dx = -rnd(0.01, 0.03)
        dy = -rnd(0.03, 0.05)
        ball = Ball(x, y, radius, dx, dy, blue)
        balls.append(ball)
        print("balls:", len(balls))
    mk_ball()

    maxv = 0.03

    person = loadsvg("person.svg")
    cvs = canvas.canvas()
    cvs.append(person)
    person = AlignBox(CanBox(cvs), "center")

    frame = 0
    while 1:
        frame += 1

        cvs = canvas.canvas()

        for ball in balls:
            ball.render(cvs)

        for ball in balls:
            ball.stroke(cvs)
            person.render(cvs, ball.x, ball.y)

        if frame > 0: # warmup
            yield cvs

        #if (frame % FPS)==0 and len(balls) < N:
        if (frame % 50)==0 and len(balls) < 4*N:
            mk_ball()

        for ball in balls:
            ball.update()
            ball.wrap(-1.1*radius, -1.1*radius, width+1.1*radius, height+1.1*radius)
            ball.fill = blue
            ball.ddx = 0.
            ball.ddy = 0.
            v = sqrt(ball.dx**2 + ball.dy**2)
            if v > maxv:
                ball.dx *= maxv/v
                ball.dy *= maxv/v

        for i in range(len(balls)):
          for j in range(len(balls)):
            if i==j:
                continue
            a = balls[i]
            b = balls[j]

            d = (b.x-a.x), (b.y-a.y)
            r = sqrt(d[0]**2 + d[1]**2)
            if r>2*radius:
                continue
            if r < 1e-2:
                continue
            r1 = r/radius
            a.ddx += -0.01*d[0]/(r1**4)
            a.ddy += -0.01*d[1]/(r1**4)

            if r<1.95*radius:
                a.fill = orange

        for ball in balls:
            if ball.dy >= 0 and ball.ddy == 0.:
                ball.ddy = -0.0001

        if len(balls) > N:
            maxv *= 0.9999
            

def ball_seq():
    import ode
    from simulate import Sphere, Sim

    mu = argv.get("mu", 0.)
    sim = Sim(mu=mu, has_gravity=False)
    sim.world.setGravity((0, -4., -1.0))

    left = -width
    right = 2*width

    walls = [
        ode.GeomPlane(sim.space, (1, 0, 0), left),
        ode.GeomPlane(sim.space, (-1, 0, 0), -right),
        ode.GeomPlane(sim.space, (0, 0, 1), 0.), # bottom
    ]

    #radius = 1.5
    radius = 0.5 * (width/10.)

#    blue = color.rgb(0.4, 0.3, 0.9, 0.8)
#    orange = color.rgb(0.8, 0.2, 0.2, 0.8)

    person = loadsvg("person.svg")
    cvs = canvas.canvas()
    cvs.append(person)
    person = AlignBox(CanBox(cvs), "center")

    balls = []
    def mkball(x, z):
        ball = Sphere(sim, radius=radius)
        ball.setPosition((x, radius, z))
        balls.append(ball)
        #print("balls:", len(balls))

    rball = lambda : mkball(
        rnd(left+radius, right-radius), height + 2*radius + 40*radius*random())

    def rball_check(x0, y0, x1, y1):
        if not balls:
            return rball()
        import kdtree
        pts = []
        for ball in balls:
            x, _, y = ball.getPosition()
            pts.append((x, y))
        tree = kdtree.create(pts)
        while 1:
            x = rnd(x0, x1)
            y = rnd(y0, y1)
            nearest = tree.search_nn_dist((x, y), (1.9*radius)**2)
            if not nearest:
                break
            #print(".", end="")
        mkball(x, y)

    nballs = argv.get("nballs", 800)
    #for i in range(400):
    #    rball_check(left+radius, radius, right-radius, 40*radius)
    for i in range(nballs):
        rball_check(left+radius, radius, right-radius, 80*radius)

    runner = sim.run()

    speed = argv.get("speed", 1)
    speed = int(speed)

    scale = 1./4

    frame = 0
    while 1:
        frame += 1

        for _ in range(speed):
            runner.__next__()

        cvs = canvas.canvas()
        if scale != 1.:
            cvs.append(trafo.scale(scale, scale, 0.5*width, 0.1*height))

        for ball in balls:
            v = ball.getPosition()
            x, _, y = v
            #if x+radius < 0. or x-radius > width:
            #    continue
            p = path.circle(x, y, radius)
            cvs.fill(p, blue)
            cvs.stroke(p)
            #person.render(cvs, x, y)

        yield cvs

        if frame % 10 == 0 and len(balls)<800:
            rball_check(left+radius, 20*radius, right-radius, 80*radius)

        if 100 < len(balls) and scale > 1./4:
            scale *= 0.999


class Live(object):

    def __init__(self, width, height, frames):
    
        import gi
        gi.require_version("Gtk", "3.0")
        from gi.repository import Gtk
        from gi.repository import GObject
    
        win = Gtk.Window()
        win.connect('destroy', self.quit)
        win.set_default_size(width, height)
        win.connect('key-press-event', self.on_key)
    
        drawingarea = Gtk.DrawingArea()
        win.add(drawingarea)
        drawingarea.connect('draw', self.render)
        self.drawingarea = drawingarea
    
        win.show_all()

        self.width = width
        self.height = height
        self.frames = frames

        GObject.timeout_add(1000./FPS, self.refresh)

        self.prev = None
        self.pause = False

        Gtk.main()

    def refresh(self):
        self.drawingarea.queue_draw()
        return True

    def quit(self, w=None):
        from gi.repository import Gtk
        Gtk.main_quit()

    def on_key(self, w, event):
        from gi.repository import Gdk
        keyval = event.keyval
        if keyval == Gdk.KEY_q:
            self.quit()
        elif keyval == Gdk.KEY_space:
            self.pause = not self.pause
        elif keyval == Gdk.KEY_Left:
            print("left!")
        elif keyval == Gdk.KEY_Right:
            print("right!")
        elif keyval == Gdk.KEY_Up:
            print("up!")
        elif keyval == Gdk.KEY_Down:
            print("down!")
        else:
            print("on_key:", event.keyval)

    def render(self, da, ctx):

        if not self.pause:
            try:
                cvs = self.frames.__next__()
            except StopIteration:
                self.quit()
                return
        else:
            cvs = self.prev
            if cvs is None:
                return

        main = setup()
        main.append(cvs)
        bound = main.get_bound_cairo()

        dx = 0 - bound.llx
        dy = self.height + bound.lly

        ctx.translate(dx, dy)
        ctx.set_line_width(_defaultlinewidth * SCALE_CM_TO_POINT)
        main.process_cairo(ctx)

        self.prev = cvs


def main(frames):

    nframes = argv.get("nframes", None)
    speed = argv.get("speed", 1)
    speed = int(speed)
    assert speed > 0

    frame = 0
    while 1:
        try:
            for _ in range(speed):
                cvs = frames.__next__()
        except StopIteration:
            break

        main = setup()
        main.append(cvs)

        name = "frames/%.4d"%frame
        main.writePNGfile("%s.png"%name)
        if frame==0:
            main.writePDFfile("%s.pdf"%name)

        print(".", end="", flush=True)
        frame += 1

        if nframes is not None and frame>nframes:
            break

    print("OK")



if __name__ == "__main__":

    live = argv.live

    frames = argv.next()
    assert frames is not None, "please specify sequence name"
    frames = eval(frames)
    frames = frames()

    if live:
        Live(W, H, frames)
    else:
        main(frames)




