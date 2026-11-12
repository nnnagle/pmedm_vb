"""Download everything the embedding needs, on a node that can reach census.gov.

The embedding's counterpart of :func:`pmedm_vb.data.prefetch.prefetch`, kept
separate so neither changes the other. Run it on a login node or dtn1/dtn2;
batch jobs then set :data:`~pmedm_vb.data.cache.OFFLINE_ENV` and read only the
cache, so anything missed here fails there at once, by name.

What is fetched, for a configuration's targets, inputs and carried ``tables``
together:

* the PUMS housing and person files and the data dictionary (the dictionary
  also serves the occupation and industry crosswalks of ``C24010``/``C24030``);
* the tract VRE files of any ``B17010A``-``I`` table, whose cells
  :func:`~pmedm_vb.assemble.constraints.race_poverty` finds by title there;
* the API cell labels of every published table (no key needed);
* with ``values=True``, every published table's estimates and MOEs at PUMA
  level from the API, which needs ``CENSUS_API_KEY`` -- these score the
  leave-one-PUMA-out check.

Every table and attribute is then built once, so a builder that needs
something not fetched fails here rather than on a compute node.
"""

from __future__ import annotations

from pathlib import Path

from pmedm_vb.assemble.constraints import RACE_POVERTY_TABLES
from pmedm_vb.data.acs_api import download_labels, download_values
from pmedm_vb.data.pums import data_dictionary, download_pums
from pmedm_vb.data.variance import download_replicates
from pmedm_vb.embed.config import EmbedConfig, resolve
from pmedm_vb.progress import stage

#: Geography of the API values fetched for scoring.
SCORING_GEOGRAPHY = "puma"


def prefetch_embedding(config: EmbedConfig, *, values: bool = True) -> list[Path]:
    """Fetch every file the configuration's tables and attributes read.

    Returns
    -------
    list of Path
        The PUMS, VRE and API files, now cached. Files already cached are
        skipped, so re-running is cheap.
    """
    area = config.area
    tables = config.table_ids
    paths: list[Path] = []
    with stage(f"prefetch embedding {config.name} ({area.slug})") as step:
        data_dictionary(area)
        paths += [download_pums(area, record_type=kind) for kind in ("housing", "person")]
        for table in tables:
            if table in RACE_POVERTY_TABLES:
                paths.append(download_replicates(area, table, geography="tract"))
        for table in tables:
            paths.append(download_labels(area, table))
        if values:
            for table in tables:
                paths.append(download_values(area, table, geography=SCORING_GEOGRAPHY))
        built = resolve(area, (*config.ids, *config.tables))
        step.detail = f"{len(paths)} files, {len(built)} tables and attributes built"
    return paths
