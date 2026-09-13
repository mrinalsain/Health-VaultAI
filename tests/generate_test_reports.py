"""
tests/generate_test_reports.py
==============================
Generates realistic sample medical reports (PDF and image formats)
for testing the extraction module:
1. lipid_panel.pdf      - Multi-page digital PDF with lipid panel results.
2. cbc.pdf              - Digital PDF with Complete Blood Count (no target values).
3. glucose_test.pdf     - Digital PDF with Fasting Glucose in mmol/L.
4. ambiguous_report.pdf - General clinical follow-up note with vitals.
5. scanned_vitals.png   - Image file to test the OCR pathway.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from PIL import Image, ImageDraw, ImageFont


def create_lipid_panel_pdf(output_path: str):
    """Creates a 2-page realistic Lipid Panel report in mg/dL."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title & Header
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12
    )
    story.append(Paragraph("MetroHealth Diagnostic Laboratory", title_style))
    story.append(Paragraph("<b>Report:</b> Comprehensive Lipid Panel", styles['Normal']))
    story.append(Paragraph("<b>Patient ID:</b> HV-90821 | <b>DOB:</b> 1978-04-12 | <b>Date:</b> 2026-08-15", styles['Normal']))
    story.append(Spacer(1, 14))

    # Lab Results Table
    data = [
        ["Analyte", "Result", "Standard Units", "Reference Range", "Flag"],
        ["Total Cholesterol", "215", "mg/dL", "< 200", "High"],
        ["HDL Cholesterol", "48", "mg/dL", "> 40", "Normal"],
        ["LDL Cholesterol", "138", "mg/dL", "< 100", "High"],
        ["Triglycerides", "145", "mg/dL", "< 150", "Normal"],
    ]
    t = Table(data, colWidths=[150, 70, 90, 110, 60])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#1A202C")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(t)
    story.append(Spacer(1, 20))
    story.append(Paragraph("<i>End of primary lab panel summary on Page 1.</i>", styles['Italic']))

    # Page Break to test multi-page extraction capability!
    story.append(PageBreak())

    # Page 2: Clinical Interpretations and Comments
    story.append(Paragraph("MetroHealth Diagnostic Laboratory — Page 2 of 2", styles['Heading3']))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Clinical Commentary:</b> Patient presents with elevated Total Cholesterol (215 mg/dL) "
        "and borderline LDL. Diet modification and cardiovascular risk stratification are advised.",
        styles['Normal']
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "<b>Vitals recorded at draw:</b> Blood Pressure: 122/80 mmHg, Pulse: 72 bpm.",
        styles['Normal']
    ))

    doc.build(story)
    print(f"Generated: {output_path}")


def create_cbc_pdf(output_path: str):
    """Creates a CBC report containing hematology terms and no target lipid/glucose values."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12
    )
    story.append(Paragraph("City Pathology Services", title_style))
    story.append(Paragraph("<b>Report:</b> Complete Blood Count (CBC) with Differential", styles['Normal']))
    story.append(Paragraph("<b>Patient ID:</b> HV-44219 | <b>Date:</b> 2026-08-18", styles['Normal']))
    story.append(Spacer(1, 14))

    data = [
        ["Test Description", "Result", "Units", "Reference Interval"],
        ["WBC (White Blood Cells)", "6.8", "x10^3/uL", "4.0 - 11.0"],
        ["RBC (Red Blood Cells)", "4.95", "x10^6/uL", "4.20 - 5.80"],
        ["Hemoglobin", "15.2", "g/dL", "13.2 - 17.5"],
        ["Hematocrit", "44.1", "%", "38.5 - 50.0"],
        ["Platelet Count", "240", "x10^3/uL", "150 - 450"],
    ]
    t = Table(data, colWidths=[180, 80, 90, 130])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#1A202C")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))
    story.append(Paragraph("<b>Morphology:</b> RBC and Platelet morphology are within normal limits.", styles['Normal']))

    doc.build(story)
    print(f"Generated: {output_path}")


def create_glucose_pdf(output_path: str):
    """Creates a Glucose test report in international mmol/L units to test unit conversion."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#1A365D"),
        spaceAfter=12
    )
    story.append(Paragraph("Apex Endocrinology & Diabetes Care", title_style))
    story.append(Paragraph("<b>Report:</b> Fasting Glucose & Glycemic Profile", styles['Normal']))
    story.append(Paragraph("<b>Patient ID:</b> HV-31902 | <b>Date:</b> 2026-08-20", styles['Normal']))
    story.append(Spacer(1, 14))

    data = [
        ["Test Name", "Result", "Units", "Reference Range"],
        ["Fasting Glucose", "5.8", "mmol/L", "3.9 - 5.5"],
        ["HbA1c", "5.6", "%", "< 5.7"],
        ["Blood Sugar Post-Meal", "6.9", "mmol/L", "< 7.8"],
    ]
    t = Table(data, colWidths=[180, 80, 90, 130])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#1A202C")),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(t)
    story.append(Spacer(1, 16))
    story.append(Paragraph("<b>Notes:</b> Fasting Blood Sugar is slightly elevated above baseline.", styles['Normal']))

    doc.build(story)
    print(f"Generated: {output_path}")


def create_ambiguous_pdf(output_path: str):
    """Creates an informal clinical note with vague keywords and vitals."""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Family Medical Associates — Outpatient Encounter", styles['Heading2']))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Date: 2026-09-02 | Patient: John Doe | Provider: Dr. R. Adams", styles['Normal']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Subjective: Patient presents for routine annual physical. No acute complaints. "
        "Reports mild work-related stress. Exercises twice weekly.",
        styles['Normal']
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Objective: Vitals taken today: Blood Pressure: 128/82 mmHg. Heart Rate: 74 bpm. "
        "Lungs clear to auscultation bilaterally. Heart rate and rhythm regular.",
        styles['Normal']
    ))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Assessment: Patient is generally well. Routine preventive screening recommended.",
        styles['Normal']
    ))

    doc.build(story)
    print(f"Generated: {output_path}")


def create_scanned_image_report(output_path: str):
    """Creates a sample image report to test the OCR parser pipeline."""
    img = Image.new('RGB', (700, 350), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    text_lines = [
        "CLINIC VITAL SIGNS & OBSERVATION SHEET",
        "Patient: Jane Smith       Date: 2026-09-05",
        "---------------------------------------------------",
        "Blood Pressure: 135/85 mmHg",
        "Pulse: 78 bpm",
        "Total Cholesterol: 198 mg/dL",
        "Glucose: 94 mg/dL",
        "---------------------------------------------------",
        "Status: Stable. Follow up in 6 months."
    ]

    y = 30
    for line in text_lines:
        draw.text((40, y), line, fill=(10, 10, 10))
        y += 30

    img.save(output_path)
    print(f"Generated: {output_path}")


def main():
    target_dir = os.path.join(os.path.dirname(__file__), "sample_reports")
    os.makedirs(target_dir, exist_ok=True)

    create_lipid_panel_pdf(os.path.join(target_dir, "lipid_panel.pdf"))
    create_cbc_pdf(os.path.join(target_dir, "cbc.pdf"))
    create_glucose_pdf(os.path.join(target_dir, "glucose_test.pdf"))
    create_ambiguous_pdf(os.path.join(target_dir, "ambiguous_report.pdf"))
    create_scanned_image_report(os.path.join(target_dir, "scanned_vitals.png"))
    print("All sample test documents generated successfully.")


if __name__ == "__main__":
    main()
