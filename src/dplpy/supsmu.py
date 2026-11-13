"""Friedman's variable-span super smoother (Friedman 1984).

A faithful numpy port of R's ``supsmu`` (the smoother dplR calls for
``method="Friedman"`` detrending), translated 1:1 from R-4.3.3
``src/library/stats/src/ppr.f`` (the ``supsmu`` and ``smooth`` subroutines and
``block data bksupsmu``). Validated against R's ``stats::supsmu`` to ~1e-9 across
cross-validated, fixed-span, and bass-enhanced settings. The sequential running
-sum loops are kept as in the FORTRAN rather than vectorised, to preserve the
exact arithmetic (and because detrending a single tree-ring series is O(n)).

References
----------
Friedman, J. H. (1984) A variable span smoother. Tech. Report LCS 5, Stanford.
"""
import numpy as np

_SPANS = (0.05, 0.2, 0.5)          # tweeter, midrange, woofer
_BIG, _SML, _EPS = 1.0e20, 1.0e-7, 1.0e-3


def _smooth(x, y, w, span, iper, vsmlsq):
    """Running-line smoother. Returns (smo, acvr). abs(iper): 1 interval, 2
    periodic. iper>0 also computes the CV |residual| into acvr."""
    n = len(x)
    smo = np.zeros(n)
    acvr = np.zeros(n)
    jper = abs(iper)
    ibw = int(0.5 * span * n + 0.5)
    if ibw < 2:
        ibw = 2
    it = 2 * ibw + 1
    if it > n:
        it = n
    xm = ym = var = cvar = fbw = 0.0
    # initial window (Fortran i=1..it)
    for i in range(1, it + 1):
        j = i
        if jper == 2:
            j = i - ibw - 1
        if j >= 1:
            xti = x[j - 1]
        else:
            j = n + j
            xti = x[j - 1] - 1.0
        wt = w[j - 1]
        fbo = fbw
        fbw = fbw + wt
        if fbw > 0.0:
            xm = (fbo * xm + wt * xti) / fbw
            ym = (fbo * ym + wt * y[j - 1]) / fbw
        tmp = 0.0
        if fbo > 0.0:
            tmp = fbw * wt * (xti - xm) / fbo
        var = var + tmp * (xti - xm)
        cvar = cvar + tmp * (y[j - 1] - ym)
    # main loop (Fortran j=1..n)
    for j in range(1, n + 1):
        out = j - ibw - 1
        inn = j + ibw
        if not (jper != 2 and (out < 1 or inn > n)):
            if out < 1:
                out = n + out
                xto = x[out - 1] - 1.0
                xti = x[inn - 1]
            elif inn > n:
                inn = inn - n
                xti = x[inn - 1] + 1.0
                xto = x[out - 1]
            else:
                xto = x[out - 1]
                xti = x[inn - 1]
            wt = w[out - 1]
            fbo = fbw
            fbw = fbw - wt
            tmp = 0.0
            if fbw > 0.0:
                tmp = fbo * wt * (xto - xm) / fbw
            var = var - tmp * (xto - xm)
            cvar = cvar - tmp * (y[out - 1] - ym)
            if fbw > 0.0:
                xm = (fbo * xm - wt * xto) / fbw
                ym = (fbo * ym - wt * y[out - 1]) / fbw
            wt = w[inn - 1]
            fbo = fbw
            fbw = fbw + wt
            if fbw > 0.0:
                xm = (fbo * xm + wt * xti) / fbw
                ym = (fbo * ym + wt * y[inn - 1]) / fbw
            tmp = 0.0
            if fbo > 0.0:
                tmp = fbw * wt * (xti - xm) / fbo
            var = var + tmp * (xti - xm)
            cvar = cvar + tmp * (y[inn - 1] - ym)
        a = 0.0
        if var > vsmlsq:
            a = cvar / var
        smo[j - 1] = a * (x[j - 1] - xm) + ym
        if iper > 0:
            h = 0.0
            if fbw > 0.0:
                h = 1.0 / fbw
            if var > vsmlsq:
                h = h + (x[j - 1] - xm) ** 2 / var
            acvr[j - 1] = 0.0
            a = 1.0 - w[j - 1] * h
            if a > 0.0:
                acvr[j - 1] = abs(y[j - 1] - smo[j - 1]) / a
            elif j > 1:
                acvr[j - 1] = acvr[j - 2]
    # recompute smo as weighted mean for non-unique x (no-op for strictly increasing x)
    j = 1
    while True:
        j0 = j
        sy = smo[j - 1] * w[j - 1]
        fbw = w[j - 1]
        if j < n:
            while x[j] <= x[j - 1]:    # x(j+1) <= x(j)
                j = j + 1
                sy = sy + w[j - 1] * smo[j - 1]
                fbw = fbw + w[j - 1]
                if j >= n:
                    break
        if j > j0:
            a = 0.0
            if fbw > 0.0:
                a = sy / fbw
            for i in range(j0, j + 1):
                smo[i - 1] = a
        j = j + 1
        if j > n:
            break
    return smo, acvr


def supsmu(x, y, w=None, span=0.0, bass=0.0, periodic=False):
    """Friedman's super smoother, faithful to R's supsmu.
    span=0.0 => cross-validated (variable) span; span>0 => fixed fraction.
    bass in (0,10] enhances smoothness under CV (bass tone control)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if w is None:
        w = np.ones(n)
    else:
        w = np.asarray(w, dtype=float)
    smo = np.zeros(n)
    iper = 2 if periodic else 1

    if x[n - 1] <= x[0]:
        sw = w.sum()
        a = (w * y).sum() / sw if sw > 0 else 0.0
        return np.full(n, a)

    i = n // 4
    j = 3 * i
    scale = x[j - 1] - x[i - 1]
    while scale <= 0.0:
        if j < n:
            j = j + 1
        if i > 1:
            i = i - 1
        scale = x[j - 1] - x[i - 1]
    vsmlsq = (_EPS * scale) ** 2
    jper = iper
    if iper == 2 and (x[0] < 0.0 or x[n - 1] > 1.0):
        jper = 1
    if jper < 1 or jper > 2:
        jper = 1

    if span > 0.0:
        smo, _ = _smooth(x, y, w, span, jper, vsmlsq)
        return smo

    # cross-validation over the three spans
    sc = np.zeros((n, 7))   # columns 1..7 -> sc[:,0..6]
    for i3 in range(1, 4):
        s_i, acvr = _smooth(x, y, w, _SPANS[i3 - 1], jper, vsmlsq)
        sc[:, 2 * i3 - 2] = s_i                 # sc(:,2i-1)
        sm_res, _ = _smooth(x, acvr, w, _SPANS[1], -jper, vsmlsq)
        sc[:, 2 * i3 - 1] = sm_res              # sc(:,2i)
    for jj in range(n):
        resmin = _BIG
        for i3 in range(1, 4):
            if sc[jj, 2 * i3 - 1] < resmin:
                resmin = sc[jj, 2 * i3 - 1]
                sc[jj, 6] = _SPANS[i3 - 1]       # sc(:,7)
        if (0.0 < bass <= 10.0) and resmin < sc[jj, 5] and resmin > 0.0:
            sc[jj, 6] = sc[jj, 6] + (_SPANS[2] - sc[jj, 6]) * \
                max(_SML, resmin / sc[jj, 5]) ** (10.0 - bass)
    sm_spans, _ = _smooth(x, sc[:, 6], w, _SPANS[1], -jper, vsmlsq)
    sc[:, 1] = sm_spans                          # sc(:,2)
    for jj in range(n):
        if sc[jj, 1] <= _SPANS[0]:
            sc[jj, 1] = _SPANS[0]
        if sc[jj, 1] >= _SPANS[2]:
            sc[jj, 1] = _SPANS[2]
        f = sc[jj, 1] - _SPANS[1]
        if f < 0.0:
            f = -f / (_SPANS[1] - _SPANS[0])
            sc[jj, 3] = (1.0 - f) * sc[jj, 2] + f * sc[jj, 0]
        else:
            f = f / (_SPANS[2] - _SPANS[1])
            sc[jj, 3] = (1.0 - f) * sc[jj, 2] + f * sc[jj, 4]
    smo, _ = _smooth(x, sc[:, 3], w, _SPANS[0], -jper, vsmlsq)
    return smo
