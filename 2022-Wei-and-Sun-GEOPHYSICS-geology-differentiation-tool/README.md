Geology Differentiation
==========================
This script allows users to interactively classify the inverted physical property values into distinct classes by creating polygons using mouse clicks. Using this script, one can reproduce the geology differentiation results shown in Fig. 5 in Wei and Sun (2022, 3D probabilistic geology differentiation based on airborne geophysics, mixed Lp norm joint inversion and petrophysical measurements).

**Authors:**
- Xiaolong Wei (xwei7@uh.edu)
- Jiajia Sun (jsun20@uh.edu)

## Requirements

- Python 3.6 or later

- SimPEG (https://simpeg.xyz/) can be installed following (https://docs.simpeg.xyz/content/basic/installing.html ).

- After successful installation of the official SimPEG, install a modified version from github (https://github.com/xiaolongw1223).

  - The modifed version of SimPEG can be installed by using pip:

        pip install git+https://github.com/xiaolongw1223/simpeg.git@Joinv_0.13.0_gzz --upgrade --user

- PyVista and PyGeo are used for 3D visualization:

  - PyVista can be installed following (https://docs.pyvista.org/getting-started/installation.html)
  - PyGeo can be installed following (https://pvgeo.org/overview/getting-started.html)

## Running codes

After unzipping the two zipped files,

- In an IPython console

      run GeologyDifferentiation.py

- In a command terminal

      $ python GeologyDifferentiation.py

## Interactive geology differentiation
- Following the prompt on the screen, input the number of geological units (each geological unit is defined by a polygon on the scatterplot of two physical property values).
- Left click the mouse to create a polygon. A red cross will be displayed after each click.
- Right click to cancel the last selected location.
- Press "enter" or "return" on the keyboard to show colorful dots in the selected polygon area.
- Repeat step 2 to create the second polygon.
- Continue until all the polygons are created.

## Example data files

- joint_dens_model_UBC.txt: jointly recovered density model shown in Fig. 4a in Wei and Sun (2022).

- joint_susc_model_UBC.txt: jointly recovered susceptibility model shown in Fig. 4b in Wei and Sun (2022).

- mesh_p4.txt: model discretization.

- mesh_p4_rm.txt: model discretization with padding cells removed.


## References
Wei, X., and J. Sun, 2022, 3D probablistic geology differentiation based on airborne geophysics, mixed Lp norm joint inversion and petrophysical measurements, JGR: Solid Earth, under review.
