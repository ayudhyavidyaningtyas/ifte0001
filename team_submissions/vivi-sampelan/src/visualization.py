"""
Chart generation for PDF reports.
"""

from typing import Dict, Optional
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from reportlab.platypus import Image
from reportlab.lib.units import inch


def create_revenue_chart(revenue_dict: Dict[str, float]) -> Optional[Image]:
    """
    Create revenue trend chart for PDF.
    
    Args:
        revenue_dict: Dictionary mapping years to revenue values (in billions)
        
    Returns:
        ReportLab Image object or None if insufficient data
    """
    if not revenue_dict or len(revenue_dict) < 2:
        return None
    
    try:
        years = sorted(revenue_dict.keys())
        values = [revenue_dict[y] for y in years]
        year_labels = [y.split('-')[0] if '-' in y else y for y in years]
        
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(year_labels, values, marker='o', linewidth=2, markersize=6, color='#2E86AB')
        ax.set_title('Revenue Trend', fontsize=11, fontweight='bold')
        ax.set_ylabel('Billions USD', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        
        return Image(buf, width=4.5*inch, height=2.2*inch)
    except Exception as e:
        print(f"⚠️ Chart generation failed: {e}")
        return None
