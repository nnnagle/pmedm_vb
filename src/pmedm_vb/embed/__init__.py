"""Stage 1 of the latent-space PMEDM: an embedding of the statewide PUMS units.

Kept apart from the PMEDM model (``assemble/``, ``solvers/``): this package
imports from there read-only -- the unit frame, the table classes, the
existing constraint and held-out builders -- and nothing there imports from
here.

* :mod:`~pmedm_vb.embed.tables` -- builders for the tables added as targets.
* :mod:`~pmedm_vb.embed.attributes` -- PUMS attributes with no published table.
* :mod:`~pmedm_vb.embed.config` -- the JSON configuration of targets and inputs.
* :mod:`~pmedm_vb.embed.prefetch` -- every download the embedding needs.
* :mod:`~pmedm_vb.embed.units` -- the target arrays, one row per unit, cached.
* :mod:`~pmedm_vb.embed.model` -- the linear decoder, the loss, whitening.
* :mod:`~pmedm_vb.embed.fit` -- the encoder-free baseline fit and its outputs.
* :mod:`~pmedm_vb.embed.check` -- check 1, reconstruction of held-out entries.
"""
