.. image:: https://gitlab.com/deltares/wflow/hydromt/badges/master/coverage.svg
   :target: https://gitlab.com/deltares/wflow/hydromt/commits/master

################################################################################
HydroMT
################################################################################

HydroMT is a python package, developed by @Deltares, to build and analysis hydrological models.
It adopts the xarray data structure for model schematization maps and pyflwdir for any
flow direction based methods.

Installation
------------

To install hydromt (add a -e flag to pip install to install developer package), do:

.. code-block:: console

  git clone https://gitlab.com/deltares/wflow/hydromt.git
  cd hydromt
  conda env create -f environment.yml
  conda activate hydromt
  pip install . 


To update hydromt, assuming you have installed using the instructions above.
Navigate to your local clone of the hydromt repository, then run the following from command line

.. code-block:: console

  conda activate hydromt
  git checkout master 
  git pull
  conda env update -f=environment.yml  # update your conda hydromt environment
  # if you have installed hydromt as a developer package (check with 'conda list hydromt') this should be sufficient otherwise do the following:
  pip uninstall hydromt
  pip install .


Documentation
-------------

`hydroMT documentation <https://deltares.gitlab.io/wflow/hydromt/>`_

License
-------

Copyright (c) 2019, Deltares

Licensed under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
