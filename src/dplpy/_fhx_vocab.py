"""Controlled vocabulary for FHX2 fire-history data.

Ported verbatim from burnr (R/sysdata.rda; Malevich, Guiterman & Margolis;
Malevich et al. 2018, Dendrochronologia). ``rec_type`` is the per-observation
record type: a fire scar or injury (by season), a recorder year, a series end
(pith/bark/inner/outer), or an estimate. Each has a single-character code used in
the ``.fhx`` file body.
"""

# The 29 canonical rec_type levels, in burnr's order (rec_type_all values).
REC_TYPE_LEVELS = [
    "null_year", "recorder_year",
    "unknown_fs", "unknown_fi", "dormant_fs", "dormant_fi",
    "early_fs", "early_fi", "middle_fs", "middle_fi",
    "late_fs", "late_fi", "latewd_fs", "latewd_fi",
    "pith_year", "bark_year", "inner_year", "outer_year", "estimate",
    "transition_fs", "transition_fi", "falldormant_fs", "falldormant_fi",
    "springdormant_fs", "springdormant_fi", "earlylw_fs", "earlylw_fi",
    "latelw_fs", "latelw_fi",
]

# rec_type name -> single-character FHX2 file code (burnr rec_type_abrv).
REC_TYPE_ABRV = {
    "null_year": ".", "recorder_year": "|",
    "unknown_fs": "U", "unknown_fi": "u", "dormant_fs": "D", "dormant_fi": "d",
    "early_fs": "E", "early_fi": "e", "middle_fs": "M", "middle_fi": "m",
    "late_fs": "L", "late_fi": "l", "latewd_fs": "A", "latewd_fi": "a",
    "pith_year": "[", "bark_year": "]", "inner_year": "{", "outer_year": "}",
    "estimate": "?",
    "transition_fs": "T", "transition_fi": "t", "falldormant_fs": "F",
    "falldormant_fi": "f", "springdormant_fs": "S", "springdormant_fi": "s",
    "earlylw_fs": "B", "earlylw_fi": "b", "latelw_fs": "C", "latelw_fi": "c",
}

# file code -> rec_type name (inverse; burnr rec_type_all / abrv2rec_type).
REC_TYPE_ALL = {code: name for name, code in REC_TYPE_ABRV.items()}

# groupings burnr keys analyses on (rec_type_scar/injury/recorder/ends).
SCAR = [
    "unknown_fs", "dormant_fs", "early_fs", "middle_fs", "late_fs", "latewd_fs",
    "transition_fs", "falldormant_fs", "springdormant_fs", "earlylw_fs", "latelw_fs",
]
INJURY = [
    "unknown_fi", "dormant_fi", "early_fi", "middle_fi", "late_fi", "latewd_fi",
    "transition_fi", "falldormant_fi", "springdormant_fi", "earlylw_fi", "latelw_fi",
]
RECORDER = ["recorder_year"] + SCAR
ENDS = ["pith_year", "bark_year", "inner_year", "outer_year"]

# rec_types that are part of the original FHX2 canon (others warn on write).
CANON = REC_TYPE_LEVELS[:18]

# characters in the file body that mean "no observation".
_NODATA = (".", "\x1a")   # null_year dot and the DOS EOF marker
