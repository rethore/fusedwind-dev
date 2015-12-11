
.. _sec_blade_geometry_ex_label:

Blade Geometry
++++++++++++++

A blade planform in FUSED-Wind is described by spanwise distributions of the variables *chord*, *twist*, *relative thickness*, and *pitch axis aft leading edge*.
The default file format used to represent a blade planform contains the following columns:

.. literalinclude:: ../fusedwind/turbine/test/data/DTU_10MW_RWT_blade_axis_prebend.dat
   :lines: 1

*x* is defied positive towards the trailing edge, *y* positive towards the blade suction side, and *z* running along the blade axis from root to tip.
Following the main axis, there are three rotations, where the blade twist is specified in *rot_z*.
The inclusion of *rot_x* and *rot_y* allows for an arbitrary orientation of the cross section.
By default *rot_x* and *rot_y* are set to zero, and it is assumed that the cross sections are locally normal to the blade axis.
Next, the *chord* and relative thickness *rthick* are specified, note that *rthick* is defined as a number in the range [0, 1].
However, the range [0, 100] is also accepted and will be normalised to 1 internally in the reader.
The pitch axis aft leading edge is the final parameter, which specifies the normalised chordwise (*x*) distance between the leading edge and the main axis.

A lofted shape can be generated from this planform in combination with a series of airfoils that are interpolated according a given interpolator, typically the relative thicknesses of the airfoils.
FUSED-Wind provides an interface to the external geometry tool called `PGL`, which optionally can be installed as a dependency when installing FUSED-Wind.
The class ``fusedwind.turbine.geometry.PGLLoftedBladeSurface`` provides an OpenMDAO interface to the ``LoftedBladeSurface`` class in `PGL`.

In the example below we show how to setup a splined blade planform and lofted blade surface. In the example, the blade planform data and the lofted surface will be discretised differently, one meant for the aerodynamic calculation, and the other for the structural.
The example is located in ``fusedwind/examples/turbine/loftedsurface_with_cps.py``.

The first step is to read in the blade planform using the :class:`fusedwind.turbine.geometry.read_bladeplanform` method.
Using the :class:`fusedwind.turbine.geometry.redistribute_planform` method, the planform is redistributed according to the desired number of points.

.. literalinclude:: ../fusedwind/examples/turbine/loftedsurface_with_cps_ex.py
    :start-after: # --- 1
    :end-before: # --- 2

Next, we add the :class:`fusedwind.turbine.geometry.SplinedBladePlanform` class is added,
which inherits from the :class:`openmdao.core.group.Group` class,
and has methods to define which planform parameters to add splines to.
:class:`fusedwind.turbine.geometry.PGLRedistributedPlanform` class is a simple class
that redistributes the planform according to another distribution, in our case
the structural discretisation.

.. literalinclude:: ../fusedwind/examples/turbine/loftedsurface_with_cps_ex.py
    :start-after: # --- 2
    :end-before: # --- 3

The next step is to define the base airfoils used to generate the lofted blade shape.
The class :class:`fusedwind.turbine.geometry.PGLLoftedBladeSurface`` provides an OpenMDAO interface to the ``LoftedBladeSurface`` class in `PGL`.
This class takes several options, which we pass as a dictionary when instantiating the class.
The base airfoils are automatically redistributed according to the same distribution
function by PGL, so the base airfoils can be distributed differently.
The `blend_var` argument tells PGL how to weight the airfoils when blending them.
By default this will be done according to relative thickness, but you can also
specify airfoils to be at specific spanwise locations resulting in interpolation
weighted according to span. See the class documentation for all the options.

.. literalinclude:: ../fusedwind/examples/turbine/loftedsurface_with_cps_ex.py
    :start-after: # --- 3
    :end-before: # --- 4

In the final step in this example we specify which parameters to add a spline to
using the `add_spline` method.
Adding a spline adds an `IndepVarComp` with an array of spline CPs, which will be named
`<varname>_C`, i.e. `chord_C` for the chord.
The `SplinedBladePlanform` class currently only supports FFD splines, but you can
choose to use either a Bezier or pchip basis spline.
The former results in smoother and more global variations where the latter
allows for local control.

.. literalinclude:: ../fusedwind/examples/turbine/loftedsurface_with_cps_ex.py
    :start-after: # --- 4
    :end-before: # --- 5

Finally, we can run the example.

.. literalinclude:: ../fusedwind/examples/turbine/loftedsurface_with_cps_ex.py
    :start-after: # --- 5
    :end-before: # --- 6

To add sweep to the blade, we need to perturb the `x_C` control points array.
The plot shows the effect on the surface.

.. literalinclude:: ../fusedwind/examples/turbine/loftedsurface_with_cps_ex.py
    :start-after: # --- 6
    :end-before: # --- 7

Modifying the chord is done in the same way by changing the `chord_C` array as shown below.

.. literalinclude:: ../fusedwind/examples/turbine/loftedsurface_with_cps_ex.py
    :start-after: # --- 7
    :end-before: # --- 8

.. _bladesurface_planform-fig:

    .. image::  images/chord.png
       :width: 49 %
    .. image::  images/twist.png
       :width: 49 %
    .. image::  images/rthick.png
       :width: 49 %
    .. image::  images/p_le.png
       :width: 49 %


.. _bladesurface_lofted-blade-fig:

.. figure:: /images/lofted_blade.png
    :width: 70 %
    :align: center

    Lofted blade shape.


.. _bladesurface_topview-fig:

.. figure:: /images/bladesurface_topview.png
    :width: 70 %
    :align: center

    Top view of lofted blade surface with sweep.


.. _bladeplanform_spline-fig:

.. figure:: /images/chord_ffd_spline.*
    :width: 80 %
    :align: center

    Blade chord pertubation.


Blade Structure Example
+++++++++++++++++++++++

The blade structure parameterization is primarily aimed for conceptual analysis and optimization,
where the geometric detail is fairly low to enable its use more efficiently in an optimization context.

A cross-section is divided into a number of *regions* that each cover a fraction of the cross-section.
Each region, in turn contains a stack of materials.
In each layer, the material type, thickness and layup angle can be specified.
The materials used in the blade are specified based on apparent properties of the constituent materials, which need to be pre-computed using simple micromechanics equations and classical lamination theory.

The figure below shows a blade cross section where the region division points (DPs) are indicated.
The location of each DP is specified as a normalized arc length along the cross section
starting at the trailing edge pressure side with a value of s=-1., and along the surface to the leading edge where s=0.,
along the suction side to the trailing edge where s=1.
Any number of regions can thus be specified, distributed arbitrarily along the surface.

.. _bladestructure_cross_sec-fig:

.. figure:: /images/cross_sec_sdef13.png
    :width: 80 %
    :align: center

    Blade cross section with region division points (DPs) indicated with red dots and shear webs drawn as green lines.

The spar caps are specified in the same way as other regions, which means that their widths and position along the chord are not default parameters.
It is only possible to place shear webs at the location of a DP, which means that a single shear web topology would require the spar cap to be split into two regions.

The full blade parameterization is a simple extrusion of the cross-sectional denition, where every region covers the entire span of the blade.
The DP curves marked with red dots in the plot below are simple 1-D arrays as function of span that as in the cross-sectional definition take a value between -1. and 1.
The distribution of material and their layup angles along the blade are also specified as simple 1-D arrays as function of span.
Often, a specific composite will not cover the entire span, and in this case the thickness of this material is simply specified to be zero at that given spanwise location.

.. _bladestructure_lofted-blade-fig:

.. figure:: /images/structural_cross_sections.png
    :width: 15cm
    :align: center

    Lofted blade with region division points indicated with red dots and shear webs drawn as green lines.

FUSED-Wind provides a simple file format for storing the structural definition of the blade.
The structure is defined in a set of files with the same *base-name*, each described below. You can find an example using this file format in fusedwind/turbine/test/data/DTU10MW*.

The *<base-name>.mat* file contains the properties of all the materials used in the blade.
The first line in the header contains the names of each of the materials.
The second line lists the names of each of the parameters defining the materials,
followed by the data with the same number of lines as materials listed in line 1 (not shown below):

.. literalinclude:: ../fusedwind/turbine/test/data/DTU10MW.mat
   :lines: 1-2

The *<base-name>.failmat* file contains the strength properties of all the materials used in the blade.
The first line in the header contains the names of each of the materials.
The second line lists the names of each of the parameters defining the strength properties of the materials,
followed by the data with the same number of lines as materials listed in line 1 (not shown below):

.. literalinclude:: ../fusedwind/turbine/test/data/DTU10MW.failmat
   :lines: 1-2

The failure criterium defined in column 1 is an integer value corresponding to
1:'maximum_strain', 2:'maximum_stress', 3:'tsai_wu'.

The *<base-name>.dp3d* file contains the DPs' positions along the span.
The header of the file contains the names of the webs in line 1, the connectivity of the shear webs with the surface DPs in the following lines, and finally the names of each of the regions.
The data (not shown below) contains in column 1, the running length along the blade axis, followed by the chordwise curve fraction of the DP along the span for each of the DPs.

.. literalinclude:: ../fusedwind/turbine/test/data/DTU10MW.dp3d
   :lines: 1-6

The *<base-name>_<rname>.st3d* files contain the thickness distributions of each of the materials in the individual regions. The header contains the following: the name of the region in line 1 and the names of each of the materials in line two.
The data (not shown below) contains in column 1, the running length along the blade axis, followed by the thicknesses of each material along the span.

.. literalinclude:: ../fusedwind/turbine/test/data/DTU10MW_region04.st3d
   :lines: 1-2



.. _bladestructure_spline-fig:

.. figure:: /images/turbine_structure_uniax_perturb.png
    :width: 80 %
    :align: center

    Blade spar cap uniax thickness pertubation.
