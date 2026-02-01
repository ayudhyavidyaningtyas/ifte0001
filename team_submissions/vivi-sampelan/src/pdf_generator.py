"""
PDF report generation module.
"""

import re
from typing import Dict, Tuple
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

from .utils import validate_ticker, get_current_date_formatted
from .data_fetcher import fetch_financial_data
from .valuation import calculate_dcf, calculate_dcf_sensitivity
from .visualization import create_revenue_chart


def md_to_paragraphs(md: str):
    """
    Minimal markdown converter for ReportLab Paragraph:
    - #, ## headings
    - - bullet
    - **bold**
    
    Args:
        md: Markdown text
        
    Returns:
        List of tuples (type, text)
    """
    lines = md.splitlines()
    blocks = []
    for line in lines:
        line = line.strip()
        if not line:
            blocks.append("")
            continue

        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)

        if line.startswith("### "):
            blocks.append(("H3", line[4:].strip()))
            continue
        if line.startswith("## "):
            blocks.append(("H2", line[3:].strip()))
            continue
        if line.startswith("# "):
            blocks.append(("H1", line[2:].strip()))
            continue
        if line.startswith("- "):
            blocks.append(("BULLET", line[2:].strip()))
            continue

        blocks.append(("P", line))

    return blocks


def generate_equity_report_pdf(
    ticker: str,
    report_text: str = None,
    output_path: str = None
) -> str:
    """
    Generate comprehensive equity research PDF report.
    
    Args:
        ticker: Stock ticker symbol
        report_text: AI-generated narrative (markdown format)
        output_path: Optional custom output path
        
    Returns:
        Path to generated PDF file
        
    Raises:
        ValueError: If data fetch or PDF generation fails
    """
    is_valid, validated_ticker = validate_ticker(ticker)
    if not is_valid:
        raise ValueError(validated_ticker)
    ticker = validated_ticker

    if output_path is None:
        output_path = f"{ticker}_equity_report.pdf"

    try:
        print(f"📊 Fetching data for {ticker}...")
        financial_data = fetch_financial_data(ticker)
        if "error" in financial_data:
            raise ValueError(f"Data fetch failed: {financial_data['error']}")

        print(f"💰 Calculating DCF...")
        dcf_data = calculate_dcf(ticker)
        if "error" in dcf_data:
            print(f"⚠️  DCF warning: {dcf_data['error']}")
            dcf_data = {}
        
        sensitivity = calculate_dcf_sensitivity(ticker)

        print(f"📄 Generating PDF...")
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=50, leftMargin=50,
            topMargin=40, bottomMargin=40
        )

        styles = getSampleStyleSheet()
        story = []

        # Define styles
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            fontSize=18,
            spaceAfter=2
        )
        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontSize=10,
            textColor=colors.grey,
            spaceAfter=10
        )
        section_header_style = ParagraphStyle(
            "SectionHeader",
            parent=styles["Heading2"],
            fontSize=11,
            fontName="Helvetica-Bold",
            spaceBefore=14,
            spaceAfter=8
        )
        metrics_style = ParagraphStyle(
            "Metrics",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.black,
            spaceBefore=6,
            spaceAfter=8
        )
        small_style = ParagraphStyle(
            "Small",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.grey
        )
        bullet_style = ParagraphStyle(
            "Bullet",
            parent=styles["Normal"],
            fontSize=9,
            leftIndent=18,
            bulletIndent=8,
            spaceAfter=3,
        )

        # Header
        company_name = financial_data.get("company_name", ticker)
        current_date = get_current_date_formatted()

        story.append(Paragraph(f"{company_name} ({ticker})", title_style))
        story.append(Paragraph(f"VIVI's EQUITY RESEARCH · {current_date}", subtitle_style))

        # Rating
        upside_pct = dcf_data.get("upside_pct", None)
        if upside_pct is None:
            rating = dcf_data.get("rating", "HOLD")
        else:
            if upside_pct > 10:
                rating = "BUY"
            elif upside_pct < -10:
                rating = "SELL"
            else:
                rating = "HOLD"

        rating_colors_map = {"BUY": colors.green, "SELL": colors.red, "HOLD": colors.orange}
        rating_color = rating_colors_map.get(rating, colors.grey)

        target_price = dcf_data.get("fair_value_per_share", "N/A")

        rating_style = ParagraphStyle(
            "Rating",
            parent=styles["Normal"],
            fontSize=12,
            textColor=rating_color,
            fontName="Helvetica-Bold",
            spaceAfter=10
        )
        story.append(Paragraph(f"{rating} · Target Price: USD {target_price}", rating_style))

        # Metrics
        current_price = financial_data.get("current_price", "N/A")
        market_cap = financial_data.get("market_cap_billions", 0)
        pe_ratio = financial_data.get("pe_ratio", "N/A")

        ebit_margin_dict = financial_data.get("ebit_margin_pct", {})
        net_margin_dict = financial_data.get("net_margin_pct", {})
        roe_dict = financial_data.get("roe_pct", {})
        ebit_dict = financial_data.get("ebit_billions", {})
        fcf_dict = financial_data.get("free_cash_flow_billions", {})
        revenue_dict = financial_data.get("revenue_billions", {})

        latest_ebit_margin = list(ebit_margin_dict.values())[0] if ebit_margin_dict else None
        latest_net_margin = list(net_margin_dict.values())[0] if net_margin_dict else None
        latest_roe = list(roe_dict.values())[0] if roe_dict else None
        latest_ebit = list(ebit_dict.values())[0] if ebit_dict else None
        latest_fcf = list(fcf_dict.values())[0] if fcf_dict else None

        ev_ebit = "N/A"
        if market_cap and latest_ebit and latest_ebit != 0:
            ev_ebit = round(market_cap / latest_ebit, 2)

        p_fcf = "N/A"
        if market_cap and latest_fcf and latest_fcf != 0:
            p_fcf = round(market_cap / latest_fcf, 2)

        metrics_text = (
            f"Share Price: USD {current_price} · Market Cap: USD {market_cap}B<br/>"
            f"PE: {pe_ratio}x · EV/EBIT: {ev_ebit}x · P/FCF: {p_fcf}x<br/>"
            f"EBIT margin: {latest_ebit_margin if latest_ebit_margin is not None else 'N/A'}% · "
            f"Net margin: {latest_net_margin if latest_net_margin is not None else 'N/A'}% · "
            f"ROE: {latest_roe if latest_roe is not None else 'N/A'}%"
        )

        story.append(Paragraph(metrics_text, metrics_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.lightgrey))
        story.append(Spacer(1, 10))

        # Latest Snapshot
        story.append(Paragraph("Latest Snapshot", section_header_style))

        netinc_dict = financial_data.get("net_income_billions", {})
        latest_revenue = list(revenue_dict.values())[0] if revenue_dict else "N/A"
        latest_netinc = list(netinc_dict.values())[0] if netinc_dict else "N/A"

        snapshot_data = [
            ["Metric", "Value"],
            ["Revenue (B)", str(latest_revenue)],
            ["EBIT (B)", str(latest_ebit) if latest_ebit is not None else "N/A"],
            ["Net Income (B)", str(latest_netinc)],
            ["ROE (%)", str(latest_roe) if latest_roe is not None else "N/A"],
        ]

        snapshot_table = Table(snapshot_data, colWidths=[2*inch, 2*inch])
        snapshot_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ]))
        story.append(snapshot_table)
        story.append(Spacer(1, 14))

        # Financial Summary
        story.append(Paragraph("Financial Summary (Last 5 FY)", section_header_style))

        fin_headers = ["Year", "Revenue (B)", "EBIT (B)", "Net Inc (B)", "FCF (B)", "EBIT %", "Net %", "ROE %"]
        fin_data = [fin_headers]

        years = sorted(revenue_dict.keys(), reverse=True) if revenue_dict else []
        for year in years[:5]:
            year_short = year.split("-")[0] if "-" in year else year
            fin_data.append([
                year_short,
                str(revenue_dict.get(year, "N/A")),
                str(ebit_dict.get(year, "N/A")) if ebit_dict else "N/A",
                str(netinc_dict.get(year, "N/A")) if netinc_dict else "N/A",
                str(fcf_dict.get(year, "N/A")) if fcf_dict else "N/A",
                str(ebit_margin_dict.get(year, "N/A")) if ebit_margin_dict else "N/A",
                str(net_margin_dict.get(year, "N/A")) if net_margin_dict else "N/A",
                str(roe_dict.get(year, "N/A")) if roe_dict else "N/A",
            ])

        fin_table = Table(
            fin_data,
            colWidths=[0.8*inch, 0.85*inch, 0.75*inch, 0.85*inch, 0.75*inch, 0.6*inch, 0.6*inch, 0.6*inch]
        )
        fin_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ]))
        story.append(fin_table)

        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Source: Company filings, yfinance; compiled {current_date}", small_style))
        story.append(Spacer(1, 16))

        # DCF Sensitivity Table
        story.append(Paragraph("DCF Sensitivity Analysis", section_header_style))

        wacc_headers = ["Terminal \\ WACC"] + [f"{w:.1f}%" for w in next(iter(sensitivity.values())).keys()]
        table_data = [wacc_headers]

        for g, row in sensitivity.items():
            table_data.append(
                [f"{g:.1f}%"] +
                [f"${row[w]:.0f}" if row[w] is not None else "N/A" for w in row]
            )

        sensitivity_table = Table(
            table_data,
            colWidths=[1.5*inch] + [1.0*inch] * (len(wacc_headers) - 1)
        )

        sensitivity_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("BACKGROUND", (0, 1), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
            ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
        ]))

        story.append(sensitivity_table)
        story.append(Spacer(1, 12))

        # Agent Narrative
        if report_text:
            for item in md_to_paragraphs(report_text):
                if item == "":
                    story.append(Spacer(1, 8))
                    continue

                kind, text = item
                
                if kind == "H1":
                    story.append(Paragraph(text, title_style))
                elif kind == "H2":
                    story.append(Paragraph(text, section_header_style))
                elif kind == "BULLET":
                    story.append(Paragraph(text, bullet_style, bulletText="•"))
                else:
                    story.append(Paragraph(text, styles["Normal"]))

        # Revenue Trend chart
        chart = create_revenue_chart(revenue_dict)
        if chart:
            story.append(Paragraph("Revenue Trend", section_header_style))
            story.append(chart)
            story.append(Spacer(1, 10))

        # Disclaimer
        story.append(Spacer(1, 16))
        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontSize=7,
            textColor=colors.grey,
            spaceBefore=12
        )
        story.append(Paragraph("Disclaimer: For educational purposes only. Not investment advice.", disclaimer_style))

        doc.build(story)
        print(f"✅ PDF generated: {output_path}")
        return output_path
    
    except Exception as e:
        raise ValueError(f"PDF generation failed: {str(e)}")
