"""Generate Level 2 (Medium) complexity charts.

Multi-problem visit with moderate scan/fax degradation.
"""

from __future__ import annotations

from _runner import parse_args, run

from src.chart_generator.image_degrader import MEDIUM_PROFILE


def main() -> None:
    args = parse_args("medium")
    run(
        level="medium",
        profile=MEDIUM_PROFILE,
        chart_kwargs={
            "n_diagnoses": 3,
            "n_procedures": 3,
            "n_medications": 5,
            "include_addendum": True,
        },
        args=args,
    )


if __name__ == "__main__":
    main()
