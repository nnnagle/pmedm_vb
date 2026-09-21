"""Turn downloaded data into the arrays the solvers consume.

:mod:`~pmedm_vb.assemble.constraints` says which published cells constrain a run
and which PUMS records feed each one,
:mod:`~pmedm_vb.assemble.design` builds the individual attribute matrices and
the zone-to-constraint aggregation operators, :mod:`~pmedm_vb.assemble.targets`
builds the constraint targets and their covariance, and
:mod:`~pmedm_vb.assemble.inputs` holds the result and pins the on-disk format.
"""
