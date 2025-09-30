"""Retail Analytics Portfolio Project.

This module implements a reproducible analytics workflow that demonstrates
proficiency with the Python data stack used by data analysts. The pipeline
performs data ingestion, cleaning, feature engineering, exploratory analysis,
advanced analytics, and report generation complete with visualizations created
with Matplotlib, Seaborn, and Plotly.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from plotly import express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class DatasetSummary:
    """Container for the key metrics used in the executive summary."""

    total_sales: float
    total_profit: float
    total_orders: int
    average_discount: float
    average_order_value: float
    repeat_purchase_rate: float

    def to_markdown(self) -> str:
        data = asdict(self)
        lines = ["| Metric | Value |", "| --- | ---: |"]
        for key, value in data.items():
            pretty_key = key.replace("_", " ").title()
            lines.append(f"| {pretty_key} | {value:,.2f} |")
        return "\n".join(lines)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the retail dataset and enforce data types."""

    df = pd.read_csv(path)
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['ship_date'] = pd.to_datetime(df['ship_date'])
    categorical_columns = ['region', 'category', 'sub_category', 'customer_segment']
    for column in categorical_columns:
        df[column] = df[column].astype('category')
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned data frame with basic quality checks applied."""

    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates(subset=['order_id', 'order_date'])
    cleaned = cleaned[(cleaned['sales'] >= 0) & (cleaned['profit'] > -1000)]
    cleaned['discount'] = cleaned['discount'].clip(0, 0.9)
    cleaned['profit'] = cleaned['profit'].fillna(0)
    cleaned['sales'] = cleaned['sales'].fillna(0)
    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based and profitability features."""

    engineered = df.copy()
    engineered['order_month'] = engineered['order_date'].dt.to_period('M').dt.to_timestamp()
    engineered['order_quarter'] = engineered['order_date'].dt.to_period('Q').dt.to_timestamp()
    engineered['order_year'] = engineered['order_date'].dt.year
    engineered['gross_margin'] = engineered.apply(
        lambda row: row['profit'] / row['sales'] if row['sales'] else 0,
        axis=1,
    )
    engineered['is_repeat_customer'] = engineered.groupby('customer_segment')['order_id'].transform(
        lambda s: s.duplicated(keep=False)
    )
    return engineered


def compute_summary(df: pd.DataFrame) -> DatasetSummary:
    total_sales = df['sales'].sum()
    total_profit = df['profit'].sum()
    total_orders = df['order_id'].nunique()
    average_discount = df['discount'].mean()
    average_order_value = total_sales / total_orders if total_orders else 0
    repeat_customers = df.groupby('customer_segment')['order_id'].nunique()
    repeat_purchase_rate = (repeat_customers[repeat_customers > 1].sum() / total_orders) if total_orders else 0
    return DatasetSummary(
        total_sales=total_sales,
        total_profit=total_profit,
        total_orders=total_orders,
        average_discount=average_discount,
        average_order_value=average_order_value,
        repeat_purchase_rate=repeat_purchase_rate,
    )


def sales_by_month(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby('order_month')
        .agg(total_sales=('sales', 'sum'), total_profit=('profit', 'sum'))
        .reset_index()
        .sort_values('order_month')
    )


def category_performance(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(['category', 'sub_category'])
        .agg(total_sales=('sales', 'sum'), avg_profit_margin=('gross_margin', 'mean'))
        .reset_index()
    )


def perform_customer_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    features = df.groupby('customer_segment').agg(
        avg_discount=('discount', 'mean'),
        avg_profit=('profit', 'mean'),
        order_frequency=('order_id', 'nunique'),
        avg_quantity=('quantity', 'mean'),
    )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = KMeans(n_clusters=min(3, len(features)), n_init=10, random_state=42)
    clusters = model.fit_predict(scaled)
    features['cluster'] = clusters
    return features.reset_index()


def create_visualizations(df: pd.DataFrame, report_dir: Path) -> Dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    output_paths: Dict[str, Path] = {}

    monthly = sales_by_month(df)
    plt.figure(figsize=(10, 6))
    plt.plot(monthly['order_month'], monthly['total_sales'], marker='o', label='Sales')
    plt.plot(monthly['order_month'], monthly['total_profit'], marker='o', label='Profit')
    plt.title('Monthly Sales & Profit Trend')
    plt.xlabel('Month')
    plt.ylabel('Amount ($)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    matplotlib_path = report_dir / 'monthly_sales_profit_matplotlib.png'
    plt.tight_layout()
    plt.savefig(matplotlib_path)
    plt.close()
    output_paths['matplotlib_monthly_trend'] = matplotlib_path

    category = category_performance(df)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=category,
        x='category',
        y='total_sales',
        hue='sub_category',
        palette='viridis'
    )
    plt.title('Sales by Category & Sub-Category')
    plt.xlabel('Category')
    plt.ylabel('Total Sales ($)')
    plt.legend(title='Sub Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    seaborn_path = report_dir / 'category_sales_seaborn.png'
    plt.tight_layout()
    plt.savefig(seaborn_path)
    plt.close()
    output_paths['seaborn_category_sales'] = seaborn_path

    segmentation = perform_customer_segmentation(df)
    fig = px.scatter(
        segmentation,
        x='avg_discount',
        y='avg_profit',
        size='order_frequency',
        color='cluster',
        hover_data=['customer_segment', 'avg_quantity'],
        title='Customer Segment Profitability Clusters',
        labels={
            'avg_discount': 'Average Discount',
            'avg_profit': 'Average Profit',
            'order_frequency': 'Number of Orders'
        },
    )
    plotly_path = report_dir / 'customer_segments_plotly.html'
    fig.write_html(plotly_path)
    output_paths['plotly_customer_segments'] = plotly_path

    return output_paths


def build_markdown_report(summary: DatasetSummary, df: pd.DataFrame, report_dir: Path, asset_paths: Dict[str, Path]) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    monthly = sales_by_month(df).tail(6)
    category = category_performance(df).sort_values('total_sales', ascending=False).head(10)

    report_lines: List[str] = [
        "# Retail Analytics Executive Summary",
        "",
        "This report provides a concise overview of commercial performance and customer behaviour.",
        "",
        "## Key Performance Indicators",
        summary.to_markdown(),
        "",
        "## Recent Monthly Performance",
        monthly.to_markdown(index=False),
        "",
        "## Top Category & Sub-Category Opportunities",
        category.to_markdown(index=False),
        "",
        "## Visual Assets",
        "- Monthly sales & profit trend (Matplotlib): " + str(asset_paths['matplotlib_monthly_trend'].name),
        "- Category sales breakdown (Seaborn): " + str(asset_paths['seaborn_category_sales'].name),
        "- Customer segmentation scatter (Plotly HTML): " + str(asset_paths['plotly_customer_segments'].name),
    ]

    report_path = report_dir / 'retail_performance_summary.md'
    report_path.write_text("\n".join(report_lines))
    return report_path


def save_summary_json(summary: DatasetSummary, path: Path) -> None:
    path.write_text(json.dumps(asdict(summary), indent=2))


def run_pipeline(data_path: Path, output_dir: Path) -> None:
    raw = load_dataset(data_path)
    cleaned = clean_data(raw)
    featured = engineer_features(cleaned)
    summary = compute_summary(featured)
    assets = create_visualizations(featured, output_dir)
    report_path = build_markdown_report(summary, featured, output_dir, assets)
    save_summary_json(summary, output_dir / 'summary.json')
    print(f"Report written to {report_path}")
    for label, path in assets.items():
        print(f"Created {label}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Retail analytics portfolio project pipeline.')
    parser.add_argument('-d', '--data', type=Path, required=True, help='Path to the retail sales CSV file.')
    parser.add_argument('-o', '--output', type=Path, default=Path('projects/retail-analytics/reports'), help='Directory to store generated reports and assets.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.data, args.output)


if __name__ == '__main__':
    main()
