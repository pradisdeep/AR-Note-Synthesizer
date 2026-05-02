"""Render a ChartData payload to a structured PDF using ReportLab.

The layout is tuned for OCR/RAG stress-testing: it includes structured
patient demographics, vitals tables, an HPI block, ROS list, diagnosis
and procedure tables, a medication list, and an assessment/plan section.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .data_generator import ChartData

PAGE_MARGIN = 0.5 * inch


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Heading1"], fontSize=14, spaceAfter=4, alignment=1
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=9, alignment=1, spaceAfter=10
        ),
        "section": ParagraphStyle(
            "section",
            parent=base["Heading2"],
            fontSize=11,
            textColor=colors.HexColor("#1a3d6b"),
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontSize=9, leading=12, spaceAfter=4
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontSize=8, leading=10
        ),
    }


def _kv_table(rows: list[tuple[str, str]], col_widths=None) -> Table:
    table = Table(rows, colWidths=col_widths or [1.2 * inch, 2.3 * inch])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return table


def _grid_table(header: list[str], rows: list[list[str]], col_widths=None) -> Table:
    data = [header] + rows
    table = Table(data, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6ecf5")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def build_pdf(chart: ChartData, pdf_path: Path) -> Path:
    """Render a chart to PDF and return the path."""

    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=LETTER,
        leftMargin=PAGE_MARGIN,
        rightMargin=PAGE_MARGIN,
        topMargin=PAGE_MARGIN,
        bottomMargin=PAGE_MARGIN,
        title="Synthetic Clinical Chart",
    )

    story = []

    # Header
    story.append(Paragraph(chart.facility["name"], styles["title"]))
    story.append(
        Paragraph(
            f"{chart.facility['address']}<br/>"
            f"Phone: {chart.facility['phone']} | Fax: {chart.facility['fax']} | "
            f"NPI: {chart.facility['npi']}",
            styles["subtitle"],
        )
    )

    # Patient + encounter side-by-side
    patient_rows = [
        ("Patient", chart.patient["name"]),
        ("DOB", f"{chart.patient['dob']} (age {chart.patient['age']})"),
        ("Sex", chart.patient["sex"]),
        ("MRN", chart.patient["mrn"]),
        ("Address", chart.patient["address"]),
        ("Phone", chart.patient["phone"]),
    ]
    encounter_rows = [
        ("Encounter", chart.encounter["datetime"]),
        ("Visit Type", chart.encounter["type"]),
        ("Account #", chart.encounter["account"]),
        ("Encounter ID", chart.encounter["encounter_id"]),
        ("Provider", chart.provider["name"]),
        ("Specialty", chart.provider["specialty"]),
    ]
    side_by_side = Table(
        [[_kv_table(patient_rows), _kv_table(encounter_rows)]],
        colWidths=[3.7 * inch, 3.7 * inch],
    )
    side_by_side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(side_by_side)
    story.append(Spacer(1, 6))

    # Insurance
    story.append(Paragraph("Insurance", styles["section"]))
    story.append(
        _kv_table(
            [
                ("Payer", chart.insurance["payer"]),
                ("Member ID", chart.insurance["member_id"]),
                ("Group", chart.insurance["group"]),
                ("Subscriber", chart.insurance["subscriber"]),
            ],
            col_widths=[1.2 * inch, 5.0 * inch],
        )
    )

    # Chief complaint
    story.append(Paragraph("Chief Complaint", styles["section"]))
    story.append(Paragraph(chart.chief_complaint, styles["body"]))

    # HPI
    story.append(Paragraph("History of Present Illness", styles["section"]))
    story.append(Paragraph(chart.hpi, styles["body"]))

    # Vitals
    v = chart.vitals
    story.append(Paragraph("Vital Signs", styles["section"]))
    story.append(
        _grid_table(
            ["BP", "HR", "RR", "Temp (F)", "SpO2", "Wt (lb)", "Ht (in)"],
            [
                [
                    f"{v['bp_sys']}/{v['bp_dia']}",
                    str(v["hr"]),
                    str(v["rr"]),
                    str(v["temp"]),
                    f"{v['spo2']}%",
                    str(v["weight_lb"]),
                    str(v["height_in"]),
                ]
            ],
            col_widths=[0.9 * inch] * 7,
        )
    )

    # ROS
    story.append(Paragraph("Review of Systems", styles["section"]))
    for line in chart.ros:
        story.append(Paragraph(line, styles["body"]))

    # PE
    story.append(Paragraph("Physical Examination", styles["section"]))
    for line in chart.physical_exam:
        story.append(Paragraph(line, styles["body"]))

    # Diagnoses
    story.append(Paragraph("Assessment / Diagnoses (ICD-10)", styles["section"]))
    story.append(
        _grid_table(
            ["ICD-10", "Description"],
            [[d["code"], d["description"]] for d in chart.diagnoses],
            col_widths=[1.0 * inch, 6.4 * inch],
        )
    )

    # Procedures
    story.append(Paragraph("Procedures / Services (CPT)", styles["section"]))
    story.append(
        _grid_table(
            ["CPT", "Description"],
            [[p["code"], p["description"]] for p in chart.procedures],
            col_widths=[1.0 * inch, 6.4 * inch],
        )
    )

    # Medications
    story.append(Paragraph("Active Medications", styles["section"]))
    for med in chart.medications:
        story.append(Paragraph(f"&bull; {med}", styles["body"]))

    # Plan
    story.append(Paragraph("Plan", styles["section"]))
    for line in chart.assessment_plan:
        story.append(Paragraph(line, styles["body"]))

    # Addenda
    for addendum in chart.addenda:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Addendum", styles["section"]))
        story.append(
            Paragraph(
                f"<b>{addendum['datetime']}</b> &mdash; {addendum['author']}",
                styles["small"],
            )
        )
        story.append(Paragraph(addendum["text"], styles["body"]))

    # Signature
    story.append(Spacer(1, 12))
    story.append(
        Paragraph(
            f"Electronically signed by {chart.provider['name']} | NPI {chart.provider['npi']}<br/>"
            f"Signed on {chart.encounter['datetime']}",
            styles["small"],
        )
    )

    doc.build(story)
    return pdf_path
