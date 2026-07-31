# Taylor-Green reference data

These datasets support the \(Re=1600\) energy/enstrophy validation plot. Files
that permit redistribution are stored without numerical modification. External
files are downloaded from their authoritative source and checked against the
SHA-256 digests in `manifest.json`.

All plotted quantities use

\[
E(t)=\frac{1}{2}\langle |u|^2\rangle,\qquad
\Omega(t)=\frac{1}{2}\langle |\nabla\times u|^2\rangle.
\]

For incompressible flow with \(\nu=1/Re\),
\(\epsilon=2\Omega/Re\). The GALAEXI file reports the incompressible
dissipation directly, so the plotting script uses
\(\Omega=Re\,\epsilon/2\).

## Sources

- `hiocfd_spectral_512.gdiag`: dealiased pseudospectral reference from HiOCFD
  case C3.5. It is not bundled because the archive does not state a
  redistribution license. Fetch and verify it with:

  ```bash
  python3 examples/turbulence/scripts/fetch_reference_data.py \
    --dataset hiocfd_spectral_512
  ```

  Cite the workshop data and van Rees et al.,
  <https://doi.org/10.1016/j.jcp.2010.11.031>.
- `opensbli_512.dat`: University of Southampton OpenSBLI dataset,
  <https://doi.org/10.5258/SOTON/401892>, Creative Commons Attribution.
- `galaexi_dg_512.csv`: GALAEXI/FLEXI validation dataset, file 291404,
  <https://doi.org/10.18419/DARUS-4139>, Creative Commons Attribution 4.0.

Reference values are rendered as discrete markers. FFTM data are rendered as
a continuous line. This makes the source of every plotted point visible and
avoids presenting a literature series as an FFTM-generated curve.

The plotting utility skips unavailable optional datasets. Explicitly requesting
an unavailable dataset fails with a command showing how to fetch it.
