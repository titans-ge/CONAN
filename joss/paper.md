---
title: 'CONAN: A Python package for modeling lightcurve and radial velocity data of exoplanetary systems'
tags:
  - Python
  - astronomy
  - exoplanet

authors:
  - name: Babatunde Akinsanmi
    orcid: 0000-0001-6519-1598
    corresponding: true # (This is how to denote the corresponding author)
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Monika Lendl
    orcid: 0000-0001-9699-1459
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Andreas Krenn
    orcid: 0000-0003-3615-4725
    affiliation: 3
  - name: Solene Ulmer-moll
    orcid: 0000-0003-2417-7006
    affiliation: 3
affiliations:
 - name: Observatoire astronomique de l’Université de Genève, chemin Pegasi 51, 1290 Versoix, Switzerland
   index: 1
 - name: Space Research Institute, Austrian Academy of Sciences, Schmiedl-strasse 6, A-8042 Graz, Austria
   index: 2
 - name: Leiden Observatory, Leiden University, Einsteinweg 55, 2333 CC, Leiden, the Netherlands
   index: 3
date: 02 May 2025
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

`CONAN` (COde for exoplaNet ANalysis) is an open-source Python package designed for comprehensive analysis of exoplanetary systems. It provides a unified Bayesian framework where diverse exoplanet datasets can be simultaneously analyzed to derive global system parameters, while allowing for an independent baseline model for each dataset. `CONAN` allows to consistently model photometric transit light curves, occultations, phase curves, and radial velocity measurements with baselines that can be any combination of parametric, sinusoidal, Gaussian Processes, and spline models.


# Statement of need
Detecting and characterising exoplanets, planets orbiting stars other than our Sun, is a major focus area of astronomical research. This endeavor increasingly relies on heterogeneous datasets spanning multiple epochs, observing techniques, instruments, and wavelenghts. Therefore, robustly estimating the physical and orbital properties of planets routinely requires simultaneous modeling of the different signals, while dealing with the unique systematics of each instrument, or the time-dependent impact of stellar activity. 

``CONAN`` meets this requirements with some of its key features outlined below: 

- Multi-dataset analysis: Seamless analysis of combined lightcurve and radial velocity datasets from various instruments.

- Multiplanet support: Simultaneously fits multiple planets in a single system.

- Comprehensive light curve modeling: Transit, occultation, and phase curve modeling including the effects such as ellipsoidal variations and Doppler beaming ([see Model definition](https://github.com/titans-ge/CONAN/wiki/LC-and-RV-models)).

- Analysis of light curve variations: including transit timing variations (TTVs) and  transit depth variations (transmission spectroscopy).

- Flexible baseline and noise modeling: Polynomial, sinusoidal, Gaussian Processes (GP), and spline functions.

- Extensible and customizable: Users can easily add new LC and RV models or modify default ones to suit their needs e.g model transit of non-spherical planets, Rossiter–McLaughlin signals, or even non-planetary signals.

- Robust Bayesian inference: Parameter estimation via MCMC (`emcee`) or nested sampling (`dynesty`)

- Compute limb darkening coefficient:  uses `ldtk` with stellar parameters to get priors for the quadratic limb darkening coefficients.

- Automated selection of parametric model parameters: Uses the Bayesian Information Criterion to suggest best combination of vectors to use in decorrelating the data

- Data download: Built-in support for downloading data from various instruments (including TESS, CHEOPS, and Kepler) and also system parameters from [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).


- Quick result visualization and manipulation: instant plot of best-fit model and result object that can be easily manipulated for customized analysis. An instant plot from a `CONAN` fit is shown in  \autoref{fig:example}.

![An example of joint fit to transit and RVs of WASP-127\,b. The top panels show the best-fit models to the ground-based (left) and TESS (right) undetrended data. The transit model is shown in red, while the detrending baseline model is shown in gold (left: parametric model; right: GP). The bottom panes shows the best-fit RV model overplotted on the detrended and phasefolded data. The details of the fit can be found in the [online documentation](https://conan-exoplanet.readthedocs.io/en/latest/tutorial/CONAN_WASP-127_LC_RV_tutorial.html#download-TESS-data).\label{fig:example}](wasp-127_joint.png)

`CONAN` was first introduced in [@Lendl2017], and has been widely used in 15 peer-reviewed publications [e.g., @Psaridi2023; @Roche2024; @Seidel2025] with a total of 419 citations.


There are similar tools to `CONAN` for performing joint fit to exoplanet data, each with its own strengths and limitations. Some of these include `Juliet`[@juliet], PyOrbit [@pyorbit], `exoplanet`[@exoplanet], `Pyaneti`[@pyaneti], `ExoFAST`[@exofast]. One of the main strengths of `CONAN`
 compared to these tools is its capability to fit a wider variety of planetary signals. None of these tools publicly available tools can model full-orbit phasecurves of exoplanets using different phase functions. Perhaps, more impressive is that `CONAN` allows the user to define the custom model they would like to use in fitting the data, opening up practically unlimited use cases for `CONAN`. Additional, `CONAN` ability to automatically select the best decorrelation vectors to use allows make it especially well-suited to modeling ground-based observations.


# Acknowledgements

We would like to thank Angelica Psaridi, Hritam Chakraborty, Dominique Petit dit de la Roche and Adrien Deline for their help in testing the `CONAN` for several use cases.`CONAN` makes use of several publicly available packages such as `emcee`[@emcee], `dynesty[@dynesty]`, `Astropy`[@astropy], `celerite`[@celerite], `spleaf`[@spleaf], `lightkurve`[@lightkurve], `numpy`[@numpy], `ldtk`[@ldtk]. We thanl the developers of these packages for their work.  

# References