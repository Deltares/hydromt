

---
title: 'HydroMT: Automated and reproducible model building and analysis'
tags:
  - Python
  - model setup
  - model analysis
  - hydrology
  - hydrodynamics
  - gis
authors:
  - name: Dirk Eilander^[Corresponding author] 
    orcid: 0000-0002-0951-8418
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
  - name: Hélène Boisgontier
    affiliation: 1
affiliations:
 - name: Deltares, The Netherlands
   index: 1
 - name: Institution for Environmental Studies (IVM), Vrije Universiteit Amsterdam, The Netherlands
   index: 2
date: 04 May 2022
bibliography: paper.bib
---

# Summary
<!-- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience. -->

HydroMT (Hydro Model Tools) is an open-source Python package that facilitates the process of 
building and analyzing spatial geoscientific models with a focus on water system models. 
It does so by automating the workflow to go from raw data to a complete model instance which 
is ready to run and to analyse model results once the simulation has finished.

# Statement of need
<!-- A Statement of Need section that clearly illustrates the research purpose of the software -->
<!-- A list of key references, including to other software addressing related needs. 
    Note that the references should include full names of venues, e.g., journals and conferences, 
    not abbreviations only understood in the context of a specific discipline. -->
<!-- Mention (if applicable) a representative set of past or ongoing research projects using the 
    software and recent scholarly publications enabled by it. -->
<!-- State of the field: Do the authors describe how this software compares to other commonly-used packages? -->

Setting up spatial geoscientific models typically requires many (manual) steps 
to process input data and might therefore be time consuming and hard to reproduce. 
Especially improving models based on global geospatial datasets, which are 
rapidly becoming available at increasingly high resolutions [@ref], might be challenging. 
Furthermore, analyzing model schematization and results from different models, 
which often use model-specific peculiar data formats, can be time consuming.

This package aims to make the model building process fast, modular and reproducible 
by configuring the model building process from a single *ini* configuration file
and model- and data-agnostic through a common model and data interface. 

# Audience 
<!-- geemap is intended for students and researchers who would like to utilize the Python ecosystem of 
    diverse libraries and tools to explore Google Earth Engine. It is also designed for
    existing GEE users who would like to transition from the GEE JavaScript API to a Python
    API. The automated JavaScript-to-Python conversion module of the geemap package can
    greatly reduce the time needed to convert existing GEE JavaScripts to Python scripts and
    Jupyter notebooks. -->


# Functionality
<!-- first page user guide -->

# Dependencies


# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements
<!-- Acknowledgement of any financial support. -->
We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References