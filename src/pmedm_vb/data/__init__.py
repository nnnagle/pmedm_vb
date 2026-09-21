"""Downloaders for the published inputs to a PMEDM problem.

:mod:`~pmedm_vb.data.cache` fetches and caches, :mod:`~pmedm_vb.data.pums`
gets the microdata, :mod:`~pmedm_vb.data.variance` gets the replicate tables
that carry both the constraint targets and their covariance,
:mod:`~pmedm_vb.data.summary` reshapes those into targets, and
:mod:`~pmedm_vb.data.geography` supplies the nesting and PUMA membership the
aggregation operators need.

``census_data_sources.md`` at the repository root records what the endpoints
look like and how that was established, so the layouts do not have to be
rediscovered. The constants here are authoritative; that file describes them.
"""
