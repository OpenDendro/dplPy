# References

The methods dplPy implements come from the published dendrochronology
literature and from four established software tools. The works below are those
cited across the dplPy source; each function's docstring names the specific
reference(s) behind it, and the [API Reference](../reference/io.md) surfaces
those docstrings directly.

## Software heritage

- **dplR** — Bunn, A.G. (2008). A dendrochronology program library in R (dplR).
  *Dendrochronologia* 26(2), 115–124. Bunn, A.G. (2010). Statistical and visual
  crossdating in R using the dplR library. *Dendrochronologia* 28(4), 251–258.
- **COFECHA** — Holmes, R.L. (1983). Computer-assisted quality control in
  tree-ring dating and measurement. *Tree-Ring Bulletin* 43, 69–78.
  Grissino-Mayer, H.D. (2001). Evaluating crossdating accuracy: a manual and
  tutorial for the computer program COFECHA. *Tree-Ring Research* 57(2), 205–221.
- **ARSTAN** — Cook, E.R. (1985). *A time series analysis approach to tree ring
  standardization.* PhD dissertation, University of Arizona.
- **CRUST** — Melvin, T.M., Briffa, K.R. (2014). CRUST: Software for the
  implementation of Regional Chronology Standardisation: Part 1. Signal-free RCS.
  *Dendrochronologia* 32(1), 7–20.

## Detrending & standardization

- Cook, E.R., Peters, K. (1981). The smoothing spline: a new approach to
  standardizing forest interior tree-ring width series for dendroclimatic
  studies. *Tree-Ring Bulletin* 41, 45–53.
- Cook, E.R., Peters, K. (1997). Calculating unbiased tree-ring indices for the
  study of climatic and environmental change. *The Holocene* 7(3), 361–370.
- Melvin, T.M. (2004). *Historical growth rates and changing climatic sensitivity
  of boreal conifers.* PhD thesis, University of East Anglia.
- Melvin, T.M., Briffa, K.R., Nicolussi, K., Grabner, M. (2007). Time-varying-
  response smoothing. *Dendrochronologia* 25(1), 65–69.
- Melvin, T.M., Briffa, K.R. (2008). A "signal-free" approach to dendroclimatic
  standardisation. *Dendrochronologia* 26(2), 71–86.
- Melvin, T.M., Briffa, K.R. (2014). CRUST (Part 1). *Dendrochronologia* 32(1),
  7–20. *(as above)*

## Chronology & signal strength

- Wigley, T.M.L., Briffa, K.R., Jones, P.D. (1984). On the average value of
  correlated time series, with applications in dendroclimatology and
  hydrometeorology. *Journal of Climate and Applied Meteorology* 23, 201–213.
  *(EPS, SSS, SNR)*
- Osborn, T.J., Briffa, K.R., Jones, P.D. (1997). Adjusting variance for
  sample-size in tree-ring chronologies and other regional-mean timeseries.
  *Dendrochronologia* 15, 89–99. *(variance stabilization)*
- Cook, E.R., Pederson, N. (2011). Uncertainty, emergence, and statistics in
  dendrochronology. In: *Dendroclimatology* (Developments in Paleoenvironmental
  Research 11), Springer, 77–112. *(effective signal, SNR)*
- Frank, D., Esper, J., Cook, E.R. (2006). Running-window rbar variance
  stabilization ("RUNNINGr"), as implemented in dplR's `chron.stabilized` — see
  the [`chron_stabilized`](../reference/chronology.md#chron_stabilized) docstring.
- Buras, A. (2017). A comment on the expressed population signal.
  *Dendrochronologia* 44, 130–132. *(SSS vs EPS)*

## Crossdating

- Baillie, M.G.L., Pilcher, J.R. (1973). A simple cross-dating program for
  tree-ring research. *Tree-Ring Bulletin* 33, 7–14. *(the t statistic)*
- Wigley, T.M.L., Jones, P.D., Briffa, K.R. (1987). Cross-dating methods in
  dendrochronology. *Journal of Archaeological Science* 14(1), 51–64.
- Fowler, A.M., Bridge, M.C. (2017). Empirically-determined statistical
  significance of the Baillie and Pilcher (1973) t statistic for British Isles
  oak. *Dendrochronologia* 42, 51–55.
- Wilson, R. (2026). Multi-parameter crossdating for sub-fossil and historical
  samples. *Dendrochronologia* 96, 126485.
- Visser, R.M. (2021). On the similarity of tree-ring patterns. *(Gleichläufigkeit
  and synchronous growth changes — see the [`glk`/`sgc`](../reference/statistics.md#glk)
  docstrings.)*

## Statistics & indices

- Biondi, F., Qeadan, F. (2008). Inequality in paleorecords. *Ecology* 89(4),
  1056–1067. *(Gini coefficient; mean-sensitivity treatment)*

## Citing dplPy

dplPy is developed by the [OpenDendro](https://opendendro.org) group (Andy Bunn,
Kevin Anchukaitis, Tyson Swetnam, and contributors). If you use it in published
work, please cite the OpenDendro project and dplPy, along with the specific
method reference(s) named in the function docstrings you relied on. See the
[project repository](https://github.com/OpenDendro/dplPy) for the current
recommended citation and version.

!!! note
    A few entries above reflect the citation as it appears in the dplPy source;
    where a volume or venue could not be confirmed, the entry points to the
    function docstring rather than assert an unverified detail — in keeping with
    the project's [fidelity principles](fidelity.md).
