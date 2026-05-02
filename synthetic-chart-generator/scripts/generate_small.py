"""Generate Level 1 (Small) complexity charts.

Single encounter with minimal diagnoses, light scan-style degradation.
"""

from __future__ import annotations

from _runner import parse_args, run

from src.chart_generator.image_degrader import SMALL_PROFILE


def main() -> None:
    args = parse_args("small")
    run(
        level="small",
        profile=SMALL_PROFILE,
        chart_kwargs={
            "n_diagnoses": 1,
            "n_procedures": 1,
            "n_medications": 2,
            "include_addendum": False,
        },
        args=args,
    )


if __name__ == "__main__":
    main()
