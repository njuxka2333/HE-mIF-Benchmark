"""Cell-niches Task 1 paired-patch extraction.

HE tissue follows CLAM ``segmentTissue`` + white/black patch checks.
mIF filtering is dataset-specific (Aligned polygon, Ultivue stitch, DAPI/CK).
"""

from .extract import FilterConfig, extract_cohort

__all__ = ["FilterConfig", "extract_cohort"]
