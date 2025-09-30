# Retail Analytics Portfolio Project

A polished, end-to-end data analytics case study that proves mastery of the
Python analytics stack. The project demonstrates the full lifecycle of a retail
sales analysis: data ingestion, cleaning, feature engineering, exploratory data
analysis, advanced analytics, and executive-ready reporting with compelling
visualizations built in Matplotlib, Seaborn, and Plotly.

## Project Highlights

- **Purpose-built dataset** – A synthetic retail dataset containing 120 orders
  across regions, product categories, and customer segments that mimics a
  real-world sales reporting environment (`data/retail_sales.csv`).
- **Robust analytics pipeline** – Modular Python code (`src/analysis.py`) that
  performs cleaning, enrichment, KPI calculation, customer segmentation with
  K-Means clustering, and generation of automated deliverables.
- **Visualization mastery** – The pipeline exports:
  - A Matplotlib trend chart for monthly sales and profit.
  - A Seaborn bar chart highlighting category and sub-category performance.
  - An interactive Plotly scatter visualizing segment profitability clusters.
- **Executive deliverables** – Auto-generated Markdown summary, JSON KPIs, and
  reusable chart assets suitable for dashboards or slide decks.

## Getting Started

1. **Create a virtual environment** (recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the analytics pipeline**:

   ```bash
   python src/analysis.py \
       --data data/retail_sales.csv \
       --output reports
   ```

   The command outputs PNG and HTML visualizations, a Markdown executive
   summary, and a JSON file with the core KPIs.

## Portfolio Talking Points

Use the project to speak about:

- Building reproducible analytics pipelines with modern Python tooling.
- Crafting stakeholder-ready insights by blending descriptive and advanced
  analytics (e.g., clustering for customer segmentation).
- Mastery of visualization libraries and communicating insights visually.
- Delivering documentation and artifacts that translate seamlessly to business
  presentations.

## Repository Structure

```
projects/retail-analytics
├── data
│   └── retail_sales.csv
├── reports
│   ├── customer_segments_plotly.html      # Generated artifact (after running)
│   ├── category_sales_seaborn.png         # Generated artifact (after running)
│   ├── monthly_sales_profit_matplotlib.png# Generated artifact (after running)
│   ├── retail_performance_summary.md      # Generated artifact (after running)
│   └── summary.json                       # Generated artifact (after running)
├── src
│   └── analysis.py
└── requirements.txt
```

> The `reports/` directory ships empty so that Git retains the folder structure.
> After executing the pipeline, it will contain the generated deliverables ready
> for portfolio sharing.

## Next Steps & Extensions

- Swap in your own transactional dataset to personalize the narrative.
- Create a Streamlit dashboard powered by the same cleaned data frame.
- Schedule the script with Airflow or GitHub Actions to emulate production
  automation.
- Translate the Markdown report into a polished PowerPoint using
  `python-pptx` or Canva for stakeholder presentations.

Showcasing this project on your resume highlights the breadth of skills needed
for a modern data analyst role, from raw data wrangling to insight delivery.
