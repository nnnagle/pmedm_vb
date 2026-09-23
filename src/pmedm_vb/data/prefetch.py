"""Download everything a run needs, and nothing else, ahead of time.

ISAAC's compute nodes cannot reach census.gov, so a batch job has to find its
inputs already cached. This fetches them from a login or data transfer node.
Files already cached are skipped (that is how :func:`~pmedm_vb.data.cache.fetch`
behaves), so re-running it is cheap, and it downloads without parsing
the large files, which is the part that belongs on a compute node.

The list mirrors what assembly reads. It is kept honest by
:data:`~pmedm_vb.data.cache.OFFLINE_ENV`: a batch job sets it, so anything
missing here fails there at once and by name rather than hanging on the
network.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pmedm_vb.config import StudyArea
from pmedm_vb.data import geography
from pmedm_vb.data.pums import data_dictionary, download_pums
from pmedm_vb.data.variance import (
    POPULATION_TABLE,
    SUMMARY_LEVELS,
    average_weight,
    download_replicates,
)
from pmedm_vb.progress import stage


def prefetch(area: StudyArea, tables: Sequence) -> list[Path]:
    """Fetch every file :func:`~pmedm_vb.assemble.build.build_all` reads.

    Parameters
    ----------
    tables:
        The :class:`~pmedm_vb.assemble.constraints.ConstraintTable` set the run
        will use; one replicate file is fetched per table and geography.

    Returns
    -------
    list of Path
        The PUMS and replicate files, now cached. The small reference files
        (data dictionary, average weights, tract-to-PUMA crosswalk) are fetched
        through the functions that read them and are not listed.
    """
    with stage(f"prefetch {area.slug}") as step:
        # Small reference files, fetched through their readers.
        data_dictionary(area)
        average_weight(area)
        geography._tract_to_puma()

        paths = [download_pums(area, record_type=kind) for kind in ("housing", "person")]
        # Replicate files are statewide, one per table and summary level. B01003
        # is read at both levels for the zero-cell model's k-values, and at
        # block group for the PUMA crosswalk.
        wanted = {(POPULATION_TABLE, level) for level in SUMMARY_LEVELS}
        wanted |= {(table.table, table.geography) for table in tables}
        for table, level in sorted(wanted):
            paths.append(download_replicates(area, table, geography=level))
        step.detail = f"{len(paths)} data files"
    return paths
