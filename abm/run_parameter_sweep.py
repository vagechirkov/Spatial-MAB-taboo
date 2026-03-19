from __future__ import annotations

"""Backward-compatible entrypoint.

Use `abm.run_slurm_jobs` for new integrations.
"""

from abm.run_slurm_jobs import main


if __name__ == "__main__":
    main()
