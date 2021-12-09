
import numpy as np
import scipy.stats as stats
import scipy.interpolate as si


def credregionML(posterior=None, percentile=0.6827, pdf=None, xpdf=None):
  """
  Compute a smoothed posterior density distribution and the minimum
  density for a given percentile of the highest posterior density.
  These outputs can be used to easily compute the HPD credible regions.
  Parameters
  ----------
  posterior: 1D float ndarray
     A posterior distribution.
  percentile: Float
     The percentile (actually the fraction) of the credible region.
     A value in the range: (0, 1).
  pdf: 1D float ndarray
     A smoothed-interpolated PDF of the posterior distribution.
  xpdf: 1D float ndarray
     The X location of the pdf values.
  Returns
  -------
  pdf: 1D float ndarray
     A smoothed-interpolated PDF of the posterior distribution.
  xpdf: 1D float ndarray
     The X location of the pdf values.
  HPDmin: Float
     The minimum density in the percentile-HPD region.
  Example
  -------
  >>> import numpy as np
  >>> npoints = 100000
  >>> posterior = np.random.normal(0, 1.0, npoints)
  >>> pdf, xpdf, HPDmin = credregion(posterior)
  >>> # 68% HPD credible-region boundaries (somewhere close to +/-1.0):
  >>> print(np.amin(xpdf[pdf>HPDmin]), np.amax(xpdf[pdf>HPDmin]))
  >>> # Re-compute HPD for the 95% (withour recomputing the PDF):
  >>> pdf, xpdf, HPDmin = credregion(pdf=pdf, xpdf=xpdf, percentile=0.9545)
  >>> print(np.amin(xpdf[pdf>HPDmin]), np.amax(xpdf[pdf>HPDmin]))
  """
  if pdf is None and xpdf is None:
    # Thin if posterior has too many samples (> 120k):
    thinning = np.amax([1, int(np.size(posterior)/120000)])
    # Compute the posterior's PDF:
    kernel = stats.gaussian_kde(posterior[::thinning])
    # Remove outliers:
    mean = np.mean(posterior)
    std  = np.std(posterior)
    k = 6
    lo = np.amax([mean-k*std, np.amin(posterior)])
    hi = np.amin([mean+k*std, np.amax(posterior)])
    # Use a Gaussian kernel density estimate to trace the PDF:
    x  = np.linspace(lo, hi, 100)
    # Interpolate-resample over finer grid (because kernel.evaluate
    #  is expensive):
    f    = si.interp1d(x, kernel.evaluate(x))
    xpdf = np.linspace(lo, hi, 3000)
    pdf  = f(xpdf)

  # Sort the PDF in descending order:
  ip = np.argsort(pdf)[::-1]
  # Sorted CDF:
  cdf = np.cumsum(pdf[ip])
  # Indices of the highest posterior density:
  iHPD = np.where(cdf >= percentile*cdf[-1])[0][0]
  mHPD = np.argmax(pdf)
  # Minimum density in the HPD region:
  HPDmin = np.amin(pdf[ip][0:iHPD])
  return pdf, xpdf, HPDmin, mHPD

