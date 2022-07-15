Mixed Lp norm joint inversion
==========================
This script implements mixed Lp norm joint inversion of gravity gradient and magnetic data. The mixed Lp norm inversion was first proposed by Fournier and Oldenburg (2019). Wei and Sun (2020) extended it to cross-gradients joint inversion. This script is based on SimPEG (Cockett et al., 2015). With this script, one can reproduce the jointly inverted density and susceptibility models shown in Fig. 4 of Wei and Sun (2022).

**Authors:**
- Xiaolong Wei (xwei7@uh.edu)
- Jiajia Sun (jsun20@uh.edu)

## Requirements

- Python 3.6 or later

- SimPEG (https://simpeg.xyz/) can be installed following (https://docs.simpeg.xyz/content/basic/installing.html ).

- After successful installation of the official SimPEG, install a modified version from github (https://github.com/xiaolongw1223). This modified version contains some changes that Xiaolong Wei made as part of his doctoral dissertation research at the University of Houston. These changes have not been merged to the official release of SimPEG yet.

  - The modifed version of SimPEG can be installed by using pip:

        pip install git+https://github.com/xiaolongw1223/simpeg.git@Joinv_0.13.0_gzz --upgrade --user

## Running codes

After unzipping the two zipped files,

- In an IPython console

      run MixedLpJointInversion.py Input.json

- In a command terminal

      $ python MixedLpJointInversion.py Input.json


## Introduction of "Input.json" file

- data_file: observed gravity gradient and magnetic data.

- mesh_file: model discretization or parameterization.

- mesh_file_rm: model discretization with padding cells removed.

- topography: topography file.

- norm_p: norm value implemented on the smallness component of the regularization term.

- norm_q: norm value implemented on the three smoothness components of the regularization term.

- alpha_s: a constant weighting parameter for smallness component.

- alpha_x: a constant weighting parameter for smoothness component in x direction.

- alpha_y: a constant weighting parameter for smoothness component in y direction.

- alpha_z: a constant weighting parameter for smoothness component in z direction.

- lamdba: weight for the cross-gradient term.

## References
Wei, X. and Sun, J., 2021. Uncertainty analysis of 3D potential-field deterministic inversion using mixed Lp norms. Geophysics, 86(6), pp.G133-G158.

Cockett, R., Kang, S., Heagy, L.J., Pidlisecky, A. and Oldenburg, D.W., 2015. SimPEG: An open source framework for simulation and gradient based parameter estimation in geophysical applications. Computers & Geosciences, 85, pp.142-154.

Fournier, D. and Oldenburg, D.W., 2019. Inversion using spatially variable mixed ℓ p norms. Geophysical Journal International, 218(1), pp.268-282.

Wei, X. and Sun, J., 2020. Uncertainty analysis of joint inversion using mixed Lp-norm regularization. In SEG Technical Program Expanded Abstracts 2020 (pp. 925-929). Society of Exploration Geophysicists.

Wei, X. and Sun, J., 2022, 3D probabilistic geology differentiation based on airborne geophysics, mixed Lp norm joint inversion and petrophysical measurements, JGR: Solid Earth, under review.
