"""Stage 2 of the latent-space PMEDM: the small-area estimator on φ.

* :mod:`~pmedm_vb.latent.model`: the model, its objective and the fit;
* :mod:`~pmedm_vb.latent.regions`: inputs for the regional run, where regions
  of PUMAs play the PUMA's role and their PUMAs the block groups';
* :mod:`~pmedm_vb.latent.scoring`: scores against the PUMS truth.

The design is in ``HANDOFF_stage2.md``. The existing PMEDM (``assemble/``,
``solvers/``) is imported, never modified.
"""
