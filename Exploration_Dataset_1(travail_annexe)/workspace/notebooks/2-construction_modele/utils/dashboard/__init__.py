from .preprocessing import (
    dashboard_notebook_preprocessing,
)

from.correlations import(
    correlation_heatmap_features_quality,
)

from .quallite_distribution import(
    quality_category_distribution,
    compare_quality_category_distribution,
)

from .quick_analyse_ocr import (
    dashboard_quality,
    dashboard_contrast_vs_quality,
    dashboard_size,
    dashboard_ratio,
    dashboard_table_segments,
    dashboard_clusters,
    dashboard_quality_quantiles_images,
)

__all__ =[
    "dashboard_notebook_preprocessing",
    "correlation_heatmap_features_quality",
    "quality_category_distribution",
    "compare_quality_category_distribution",
    "dashboard_quality",
    "dashboard_contrast_vs_quality",
    "dashboard_size",
    "dashboard_ratio",
    "dashboard_table_segments",
    "dashboard_clusters",
    "dashboard_quality_quantiles_images",
]
