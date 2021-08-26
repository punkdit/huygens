
<img src="images/Christiaan_Huygens.png" align="right" />

huygens
=======

This is a python package for drawing diagrams.
Intended to have multiple backends, currently the only
backend implemented uses [cairo](https://www.cairographics.org/).
Perhaps there will also be a [PyX](https://pyx-project.org/) backend, or a 
[TikZ](https://ctan.org/pkg/pgf?lang=en) backend.

The huygens package can also include text from $\TeX$ via pdftex, pdflatex, xetex or xelatex.

The [huygens.box](test_box.html) and [huygens.diagram](test_diagram.html) modules build on the basic graphics primitives
to define composable elements.
Geometry and layout of these elements is generated using a linear constraint solver.
Therefore, this system becomes a declarative graphics layout package.

The huygens package is at an early stage of development, so expect
things to change drastically and without warning.

User guide
----------

__[huygens](test_canvas.html)__
Basic drawing functions. Lines, curves, text and so on.

__[huygens.turtle](test_turtle.html)__
A simple Turtle class for sequentially building paths.
Also fun.

__[huygens.sat](test_sat.html)__
A convenient interface to a linear programming constraint solver. 
Currently uses 
[scipy.optimize.linprog](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html)

__[huygens.box](test_box.html)__
Structure figures into rectangular `Box`s.
Various ways of combining these: `HBox`, `VBox`, `TableBox`, etc.
The layout is determined by constraints, and so this uses
the `huygens.sat` module.

__[huygens.diagram](test_diagram.html)__
Building on the `box` module to make string diagrams.

See also
--------

Some related projects:

[Foundations of brick diagrams](https://arxiv.org/abs/1908.10660)
Describes horizontal and vertical composition of string diagrams.

[diagrams](https://archives.haskell.org/projects.haskell.org/diagrams/)
A declarative, domain-specific language written in Haskell.

[Compose.jl](https://github.com/GiovineItalia/Compose.jl)
A declarative vector graphics library for Julia.

[Experiments in Constraint-based Graphic Design](https://www.anishathalye.com/2019/12/12/constraint-based-graphic-design/#case-studies) 
Describes a DSL called "Basalt" written in Python, based on
the logic of non-linear real arithmetic, which is an
extension of linear programming. No code available.

[Penrose](http://penrose.ink) 
"Create beautiful diagrams just by typing mathematical notation in plain text."
[Source here](https://github.com/penrose/penrose)
Written in Haskell.  Users not welcome (yet).

[Graphviz](https://graphviz.org/)
Declarative language for rendering heirarchical data structures.
[Apparently](https://news.ycombinator.com/item?id=23477034)
"20 year old code that was basically a prototype that escaped from the lab".
See also 
[this blog post.](https://ncona.com/2020/06/create-diagrams-with-code-using-graphviz/)


[TikZ](https://ctan.org/pkg/pgf?lang=en)
Possibly the most widely used package for creating
graphics for integration with latex documents.

[Asymptote](https://asymptote.sourceforge.io/)
A custom language for doing vector graphics inspired by MetaPost.

[manim](https://github.com/3b1b/manim)
Primarily used for animations, this library also has
an interface to latex. It also is a python library using pycairo.



