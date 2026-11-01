# Climate response

Correlation of a tree-ring chronology with monthly and seasonal climate
(SEASCORR; Meko, Touchan & Anchukaitis 2011), bootstrapped monthly correlation
and response functions (`climate_response`; Fritts et al. 1971, treeclim), the
**stability** of a correlation through time by a moving-window analysis with
Monte-Carlo significance (`moving_corr`), superposed epoch analysis of event
responses (`sea`), and reconstruction of climate from a tree-ring network by
orthogonal spatial / principal-components regression (OSR; Cook, Briffa & Jones
1994). See the [Climate–growth response](../guide/seascorr.md),
[Correlation and response functions](../guide/climate-response.md),
[Moving correlation](../guide/moving-correlation.md) and
[Superposed epoch analysis](../guide/superposed-epoch.md) guides for worked
examples, and [Reconstruction validation](../background/reconstruction-validation.md)
for how `pcreg` validates a reconstruction and what each statistic in
`result.stats` means.

## seascorr

::: dplpy.climate.seascorr

## SeascorrResult

::: dplpy.climate.SeascorrResult

## climate_response

::: dplpy.climate_response.climate_response

## ClimateResponseResult

::: dplpy.climate_response.ClimateResponseResult

## pcreg

::: dplpy.pcreg.pcreg

## PcregResult

::: dplpy.pcreg.PcregResult

## sea

::: dplpy.sea.sea

## SeaResult

::: dplpy.sea.SeaResult

## moving_corr

::: dplpy.moving_corr.moving_corr

## MovingCorrResult

::: dplpy.moving_corr.MovingCorrResult

## MovingCorrGridResult

::: dplpy.moving_corr.MovingCorrGridResult
