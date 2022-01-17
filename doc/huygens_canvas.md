# Canvas

The canvas is a front-end drawing API modelled after the amazing PyX package.

The canvas coordinates are positive in the upper right quadrant.


```python
from huygens import canvas, path, color

cvs = canvas.canvas()
cvs.stroke(path.line(0., 0., 3., 2.))
cvs.fill(path.circle(3., 2., 0.2), [color.rgb.red])
```




![svg](output_1_0.svg)




```python
# The canvas also has a writePDFfile method.
# Unlike PyX, angles are specified in radians in calls to path.arc and trafo.rotate.

from math import pi
from huygens import style, trafo, linestyle

cvs = canvas.canvas()

cvs.stroke(path.circle(0., 0., 1.), 
    [style.linewidth.thick, color.rgb.blue, linestyle.dashed])
cvs.text(0., 0., "hey there!", [trafo.rotate(0.5*pi)])

```




![svg](output_2_0.svg)




```python
# Composite paths are built using the path.path constructor:

cvs = canvas.canvas()
p = path.path([ 
    path.moveto(0., 0.),
    path.arc(0., 0., 1., 0., 0.5*pi),
    path.lineto(-1., 1.), path.arc(-1., 0., 1., 0.5*pi, 1.0*pi),
    path.arc(-1.5, 0., 0.5, 1.0*pi, 2.0*pi), path.closepath() ])

cvs.fill(p, [color.rgb.red, trafo.scale(1.2, 1.2)])
cvs.stroke(p, [color.rgb.black, style.linewidth.THick])
```




![svg](output_3_0.svg)




```python
# Building these paths are easier using the turtle module.
```
