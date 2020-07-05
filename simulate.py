#!/usr/bin/env python3

"""
Simulate rolling dice, etc.

Requires:
https://github.com/filipeabperes/Py3ODE

Ode manual:
http://ode.org/wiki/index.php?title=Manual

Howto:
http://ode.org/wiki/index.php?title=HOWTO
"""

import sys, os, random, time
from math import *
from random import random, shuffle
from functools import reduce
from operator import add

import ode

from bruhat.argv import argv


class Vec(object):
    def __init__(self, x=0, y=0, z=0):
        self.v = (1.*x, 1.*y, 1.*z)
    def __str__(self):
        return str(self.v)
    __repr__ = __str__
    def __add__(self, other):
        v = self.v
        u = other.v
        return Vec(v[0]+u[0], v[1]+u[1], v[2]+u[2])
    def __sub__(self, other):
        v = self.v
        u = other.v
        return Vec(v[0]-u[0], v[1]-u[1], v[2]-u[2])
    def __neg__(self):
        v = self.v
        return Vec(-v[0], -v[1], -v[2])
    def __rmul__(self, r):
        v = self.v
        return Vec(r*v[0], r*v[1], r*v[2])
    def __floordiv__(self, r):
        #print("__rfloordiv__")
        return self.__rmul__(1./r)
    def __truediv__(self, r):
        #print("__rtruediv__")
        return self.__rmul__(1./r)
    def __getitem__(self, idx):
        return self.v[idx]
    @classmethod
    def promote(cls, item):
        if isinstance(item, Vec):
            return item
        return Vec(*item)


class Item(object):
    def __init__(self, sim, label=None):
        self.sim = sim
        self.label = label
        sim.items.append(self)

    def on_sim(self):
        pass

    def serialize(self):
        data = {}
        if self.label is not None:
            data["label"] = self.label
        return data


# ----------------------------------------------------------------


class BodyItem(Item):
    """
    methods:
        addForce addForceAtPos addForceAtRelPos addRelForce addRelForceAtPos
        addRelForceAtRelPos addRelTorque addTorque disable enable
        getAngularVel getFiniteRotationAxis getFiniteRotationMode
        getForce getGravityMode getLinearVel getMass getNumJoints
        getPointVel getPosRelPoint getPosition getQuaternion
        getRelPointPos getRelPointVel getRotation getTorque isEnabled
        isKinematic setAngularVel setDynamic setFiniteRotationAxis
        setFiniteRotationMode setForce setGravityMode setKinematic
        setLinearVel setMass setMovedCallback setPosition setQuaternion
        setRotation setTorque vectorFromWorld vectorToWorld
    """

    size = None
    lookup = {}

    def __init__(self, sim, body, geom, label=None, color=(1., 1., 1.), **kw):
        Item.__init__(self, sim, label, **kw)
        self.body = body
        self.geom = geom
        self.color = color
        BodyItem.lookup[geom] = self

    def __str__(self):
        return self.label or "BodyItem(...)"

    def __getattr__(self, attr):
        return getattr(self.body, attr)

    def on_collision(self, other, depth):
        pass

    def getPosition(self):
        return Vec(*self.body.getPosition())
    
    def getForce(self):
        return Vec(*self.body.getForce())

    #def addForce
    
    def serialize(self):
        body = self.body
        x,y,z = body.getPosition()
        R = body.getRotation()
        rot = [R[0], R[3], R[6], 0.,
               R[1], R[4], R[7], 0.,
               R[2], R[5], R[8], 0.,
               x, y, z, 1.0]
        #print("frame", frame)
    
        data = Item.serialize(self)
        data.update({
            "class" : self.__class__.__name__,
            "pos" : (x, y, z),
            "rot" : R,
            "size" : self.size,
            "color" : self.color,
        })
        return data


    _fixed_joint = None
    def set_fixed(self):
        sim = self.sim
        joint = ode.FixedJoint(sim.world)
        joint.attach(self.body, None)
        joint.setFixed() 
        self._fixed_joint = joint

    def is_fixed(self):
        return self._fixed_joint is not None



class Box(BodyItem):
    def __init__(self, sim, density=1000, lx=1.0, ly=1.0, lz=1.0, label=None, **kw):
        body = ode.Body(sim.world)
        M = ode.Mass()
        M.setBox(density, lx, ly, lz)
        body.setMass(M)
    
        geom = ode.GeomBox(sim.space, lengths=(lx, ly, lz))
        geom.setBody(body)

        self.size = (lx, ly, lz)
        BodyItem.__init__(self, sim, body, geom, label, **kw)


class Cylinder(BodyItem):
    def __init__(self, sim, density=1000, direction=3, radius=1.0, length=1.0, label=None, **kw):
        # direction: 1=x, 2=y, 3=z
        body = ode.Body(sim.world)
        M = ode.Mass()
        M.setCylinder(density, direction, radius, length)
        body.setMass(M)
    
        geom = ode.GeomCylinder(sim.space, radius, length)
        geom.setBody(body)

        self.size = (radius, radius, length)
        BodyItem.__init__(self, sim, body, geom, label, **kw)


class Dice(Box):
    pass


class Domino(Box):
    pass


class Sphere(BodyItem):
    def __init__(self, sim, density=1000, radius=1.0, label=None, **kw):
        body = ode.Body(sim.world)
        M = ode.Mass()
        M.setSphere(density, radius)
        self.size = (radius, radius, radius) # ?
        body.setMass(M)
    
        geom = ode.GeomSphere(sim.space, radius)
        geom.setBody(body)

        BodyItem.__init__(self, sim, body, geom, label, **kw)


# ----------------------------------------------------------------


class Camera(Item):
    def __init__(self, sim, location, look_at, aspect, angle=45.):
        Item.__init__(self, sim)
        self.location = Vec.promote(location)
        self.look_at = Vec.promote(look_at)
        self.aspect = aspect
        self.angle = angle
        self.attrs = {}

    def serialize(self):
        data = {
            "class" : self.__class__.__name__,
            "location" : self.location,
            "look_at" : self.look_at,
            "right" : (-self.aspect, 0., 0), # for povray
            "angle" : self.angle,
        }
        data.update(self.attrs)
        return data

    def __setitem__(self, key, value):
        self.attrs[key] = value


class Sim(object):
    
    def __init__(self, dt=argv.get("dt", 0.02), mu=5000, 
            floor_mu=None, bounce=1.2, 
            has_floor=True, has_gravity=True,
            threshold=0.001, ERP=0.8, CFM=1e-5):
        world = ode.World()

        if has_gravity:
            world.setGravity((0, -9.81, 0))
        world.setERP(ERP)
        world.setCFM(CFM)
        #world.setCFM(1e-4) # more spongey 
        space = ode.Space()
        
        if has_floor:
            self.floor = ode.GeomPlane(space, (0, 1, 0), 0)
        else:
            self.floor = None
        
        self.world = world
        self.space = space
        self.items = []
        self.transient = []
        self.dt = dt
        self.mu = mu
        self.threshold = threshold
        if floor_mu is None:
            floor_mu = mu
        self.floor_mu = floor_mu
        self.bounce = bounce

    def serialize(self):
        line = []
        for idx, body in enumerate(self.items):
            data = body.serialize()
            line.append(data)
        for item in self.transient:
            line.append(item)
        del self.transient[:]
        return line

    def remove(self, body):
        assert body in self.items
        self.items.remove(body)
        self.space.remove(body.geom)
    
    def on_collision(self, args, geom1, geom2):
        contacts = ode.collide(geom1, geom2)
        world, contactgroup = args
        #print("[%d]"%len(contacts), end="")
        threshold = self.threshold
        for c in contacts:
            c.setBounce(self.bounce)
            #u, v = c.getMotion1(), c.getMotion2()
            #if abs(u)>EPSILON or abs(v)>EPSILON:
            #    print(u, v)
            mu = self.mu
            if isinstance(geom1, ode.GeomPlane) or isinstance(geom2, ode.GeomPlane):
                mu = self.floor_mu
            else:
                (pos, normal, depth, _geom1, _geom2) = c.getContactGeomParams()
                bi1 = BodyItem.lookup[geom1]
                bi2 = BodyItem.lookup[geom2]
                bi1.on_collision(bi2, depth)
                bi2.on_collision(bi1, depth)
                if abs(depth)>threshold:
                    #print("%.4f %.6f %s %s"%(self.t, depth, bi1, bi2))
                    item = {"class":"Collision", "t":self.t, "depth":depth, "left":bi1.label, "right":bi2.label}
                    self.transient.append(item)
                    threshold = 1e10 # only output one collision here...

            c.setMu(mu)
            j = ode.ContactJoint(world, contactgroup, c)
            b1 = geom1.getBody()
            b2 = geom2.getBody()
            j.attach(b1, b2)
            #print(b1, b2, file=sys.stderr)
            #print(geom1, geom2, file=sys.stderr)
            #print(isinstance(geom1, ode.GeomBox), file=sys.stderr)
    
    def run(self, nframes=None, top=1, bot=2, dt=None):
        world = self.world
        space = self.space

        contactgroup = ode.JointGroup()
        frame = 0
        if dt is not None:
            self.dt = dt
        self.t = 0.
        
        running = True
        while running:
        
            line = self.serialize()
            yield line
        
            frame += 1
        
            for i in range(top):
                space.collide((world, contactgroup), self.on_collision)
                world.step(self.dt/bot)
                self.t += self.dt/bot
                contactgroup.empty()

            for body in self.items:
                body.on_sim()

            #print(".", end="", flush=True, file=sys.stderr)
        
            if nframes is not None and frame >= nframes:
                break

        print("scale = %.4f" % (bot / (25.*self.dt*top)), file=sys.stderr)
    

    
def sim_single():
    ndice = argv.get("ndice", 1)
    velocity = argv.get("velocity", 5.)

    sim = Sim(mu=1e5)

    delta = Vec(5., 5., 5.)
    v0 = Vec(0., 7., 0.) # look_at
    camera = Camera(sim, v0+delta, v0, 5./3) 

    for idx in range(ndice):
        body = Dice(sim)
        body.setPosition(v0)
        theta = 0.7*pi
        a = cos(theta)
        b = sin(theta)
        body.setRotation([a, 1.5, -b, -1., 1., 0., b, 0., a])
        #body.addForce([-0.0, 0.1, 0])
        #body.setAngularVel((0., 0., 0.))
        body.setLinearVel((velocity, 0, 0))

    nframes = argv.get("nframes", 600)

    r = 0.9
    for frame, line in enumerate(sim.run(nframes)):
        print(line)

        v0 = r*camera.look_at + (1.-r)*body.getPosition()
        camera.look_at = v0
        camera.location = v0+delta
        #camera.location = (1.-r) * v0 + r*v1
        #camera.look_at = (1.-r) * look_at_0 + r*look_at_1

EPSILON = 1e-4

def sim_balls():
    velocity = argv.get("velocity", 5.)

    sim = Sim(mu=1e4, bounce=1.0)

    labels = ["ball%d"%i for i in range(1, 16)]
    shuffle(labels)

    #body = Sphere(sim, label="ball1")
    #body.setPosition((0., 1+EPSILON, 0.))

    R = 2. + EPSILON
    v0 = Vec(0., 1. + EPSILON, 0.)

    theta = 0.
    w0 = Vec(R*sin(theta), 0., R*cos(theta))
    theta += 2*pi / 6
    w1 = Vec(R*sin(theta), 0., R*cos(theta))

    count = 0
    balls = []
    for i in range(5):
      for j in range(5):
        if i+j > 4:
            continue
        v = v0 + (i-1.5)*w0 + (j-1.5)*w1
        label = labels[count]
        body = Sphere(sim, label=label)
        body.setPosition(v)

        w = 2*pi*random()
        a, b = cos(w), sin(w)
        body.setRotation([a, 1.5, -b, -1., 1., 0., b, 0., a])

        balls.append(body)

        count += 1

    delta = Vec(20.2, 20., 20.2)
    look_at = Vec(0., 0., 0.) # look_at
    camera = Camera(sim, look_at+delta, look_at, 5./3) 

    i = sim.run()
    for count in range(400):
        for b in balls:
            v = b.getPosition()
            #print(v)
            f = -10000. * v
            b.addForce(f)
            #print(b.getForce())
        line = i.__next__()
        #print(line)
    #assert 0
    #return

    if 1:
        v0 = Vec(12., 1.+EPSILON, -0.0)
        body = Sphere(sim, label="ball0")
        body.setPosition(v0)
        body.setLinearVel((-50., 0, 0))

    nframes = argv.get("nframes", 600)

    r = 0.9
    #for frame, line in enumerate(sim.run(nframes)):
    for count in range(nframes):
        line = i.__next__()
        print(line)



def sim_pair():
    ndice = argv.get("ndice", 2)
    velocity = argv.get("velocity", 0.)

    sim = Sim()

    look_at_0 = Vec(0., 5., 0.)
    look_at_1 = Vec(2.5, 0.0, 0.) 
#    v0 = Vec(5., 11.5, 13.)
    v0 = Vec(10., 11.5, 2.)
    v1 = Vec(5., 18., 23.)
    camera = Camera(sim, v0, look_at_0, 5./3) 

    if argv.blur:
        camera["aperture"] = 0.4
        camera["blur_samples"] = 100
        camera["focal_point"] = (0, 0, 0)
        
    for idx in range(ndice):
        body = Dice(sim)
        body.setPosition((1.0*idx, 5.+1.5*idx, 0.,))
        theta = 0.7*pi + 1.3*idx
        a = cos(theta)
        b = sin(theta)
        body.setRotation([a, 1.5, -b, -1., 1., 0., b, 0., a])
        #body.addForce([-0.0, 0.1, 0])
        #body.setAngularVel((0., 0., 0.))
        body.setLinearVel((velocity, 0, 0))

    nframes = argv.get("nframes", 600)

    for frame, line in enumerate(sim.run(nframes)):
        print(line)

        r = min(1., (1.2 * frame / nframes)) # 0 .. 1
        r = 0.5 - 0.5*cos(r*pi)
        camera.location = (1.-r) * v0 + r*v1

        r = min(1., (2.0 * frame / nframes)) # 0 .. 1
        r = 0.5 - 0.5*cos(r*pi)
        camera.look_at = (1.-r) * look_at_0 + r*look_at_1
        #v = (5., v[1]+0.01, v[2])


def old_sim_multi():
    liness = []
    n = None
    ndice = 5
    nframes = argv.get("nframes", 600)
    for i in range(ndice):
        sim = Sim()

        if i==0:
            camera = Camera(sim,
                (5., 10., 15.), # location
                (2.5, 3.0, 0.), # look_at
                5./3) # aspect
        
        offset = 1.*i/ndice

        body = Dice(sim)
        body.setPosition((
            - 0.0, 
            6. + 0.0*offset, # height
            0.*offset,
        ))
        theta = 0.7*pi
        a = cos(theta)
        b = sin(theta)
        body.setRotation([a, 1.5, -b, -1., 1., 0., b, 0., a])
        #body.addForce([-0.0, 0.1, 0])
        #body.setLinearVel((4.0 + 1.0*offset, 0., 0.))
    
        body.setAngularVel((0.4*offset, 0., 0.))
    
        theta = 2*pi*offset
        dx = 0.5*sin(theta)
        dy = 0.5*cos(theta)
        body.setLinearVel((1.0+dx, 0, dy))

        lines = list(sim.run(nframes))
        n = len(lines)
        liness.append(lines)

    for i in range(n):
        line = []
        for _line in liness:
            line += _line[i]
        print(line)


def sim_multi():
    liness = []
    n = None
    ndice = 5
    nframes = argv.get("nframes", 600)
    dt = argv.get("dt", 0.02)

    x = Vec(0, 6, 0)
    #location = Vec(5., 10., 15.)
    #look_at = Vec(2.5, 3., 0.)
    #delta = location - look_at
    #print(delta)

    # negative z moves cam to the right (scene goes left)
    # positive z moves cam to the left (scene goes right)
    start = Vec(6., 6.5, +2)
    stop = Vec(8., 4., +4.)

    delta = Vec(4., 3., 4.)

    sims = []
    dice = []
    for i in range(ndice):
        sim = Sim()

        if i==0:
            camera = Camera(sim, start+delta, start, 5./3)
        sims.append(sim)
        

    def make_dice():
        i = len(dice)
        assert i < ndice
        sim = sims[i]
        offset = 1.*i/ndice

        body = Dice(sim)
        body.setPosition(x)
        theta = 0.7*pi
        a = cos(theta)
        b = sin(theta)
        body.setRotation([a, 1.5, -b, -1., 1., 0., b, 0., a])
        #body.addForce([-0.0, 0.1, 0])
        #body.setLinearVel((4.0 + 1.0*offset, 0., 0.))
    
        body.setAngularVel((0.4*offset, 0., 0.))
    
        theta = 2*pi*offset
        dx = 0.5*sin(theta)
        dy = 0.5*cos(theta)
        body.setLinearVel((1.0+dx, 0, dy))
        dice.append(body)

    dt = 0.02
    iters = [sim.run(dt=dt) for sim in sims]

    #camera.look_at = stop
    #camera.location = stop+delta

    for frame in range(nframes):

        if len(dice) < ndice and frame % 15 == 0:
            make_dice()

#        x0 = Vec(0, 0, 0)
#        for i in range(ndice):
#            pos = dice[i].getPosition()
#            x += pos
#        x = x/ndice
        #r = 0.01
        #x = (1.-r)*camera.look_at + r*Vec()

#        v = camera.look_at
#        if v[1] > 1.:
#            v = v + 0.7*dt*Vec(1, -1, 0)
#        camera.look_at = v
#        camera.location = v+delta

        r = (frame-1)/nframes # 0 ... 1
        r = 0.5 - 0.5*cos(pi*r)

        v = (1-r)*start + r*stop
        camera.look_at = v
        camera.location = v+delta

        line = []
        for i in range(ndice):
            sim = sims[i]
            line += iters[i].__next__()
        print(line)





if __name__ == "__main__":

    name = argv.next()

    if name is not None:
        fn = eval(name)
        fn()

    #main()
    #main_multi()



