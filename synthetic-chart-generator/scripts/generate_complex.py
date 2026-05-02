"""Generate Level 3 (Complex) complexity charts.

Heavy comorbidity load and aggressive fax-style degradation. Use this set
to stress-test OCR error rates and downstream RAG chunking quality.
"""

from __future__ import annotations

from _runner import parse_args, run

from src.chart_generator.image_degrader import COMPLEX_PROFILE


def main() -> None:
    args = parse_args("complex")
    run(
        level="complex",
        profile=COMPLEX_PROFILE,
        chart_kwargs={
            "n_diagnoses": 6,
            "n_procedures": 5,
            "n_medications": 8,
            "include_addendum": True,
            "n_noise_sections": 3,
        },
        args=args,
    )


if __name__ == "__main__":
    main()
