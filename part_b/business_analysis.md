# Business Case Analysis — Promotion Effectiveness at a Fashion Retail Chain

---

## B1. Problem Formulation

### B1(a) — ML Problem Formulation (3 marks)

**Target Variable:** `items_sold` — the number of items sold in a store during a given month.

**Candidate Input Features:**
- Store-level: `store_size`, `location_type` (urban/semi-urban/rural), 
  `monthly_footfall`, `competition_density`
- Promotion-level: `promotion_type` (Flat Discount, BOGO, Free Gift, 
  Category-Specific, Loyalty Points)
- Calendar features: `month`, `is_weekend`, `is_festival`, `is_month_end`
- Historical: lagged sales from the prior month (if available)

**Type of ML Problem:** This is a **supervised regression** problem.

**Justification:** The target variable (`items_sold`) is a continuous numeric 
value — we want to predict the exact quantity, not a category. Regression 
is appropriate when the output is a number on a continuous scale. The 
historical data provides labelled examples (store × month × promotion → 
actual items sold), making supervised learning the correct paradigm.

---

### B1(b) — Why Items Sold Over Revenue (3 marks)

**Items sold is a more reliable target because:**

1. **Revenue varies with price:** A store running a 50% Flat Discount may 
   generate lower revenue even when it sells more items. Revenue confounds 
   promotion effectiveness with pricing decisions.

2. **Volume captures true demand response:** We want to know which promotion 
   drives customer behaviour — not which one preserves the highest margin.

3. **Stability:** Revenue can spike or drop due to product mix changes 
   (e.g., one expensive item sold) whereas item count is more stable 
   and consistent across stores.

**Broader Principle:** Target variable selection should align with the 
**true business question** rather than the metric that's easiest to record. 
A technically perfect model optimising the wrong target is worse than a 
simpler model with the right one. This is sometimes called "Goodhart's Law" 
— when a measure becomes a target, it ceases to be a good measure.

---

### B1(c) — Against a Single Global Model (2 marks)

**Problem with one global model:** A rural store's customers respond 
differently to promotions than urban ones. A BOGO deal might work well 
in urban areas with high footfall but fail in rural stores with different 
demographics and purchasing power. Pooling all data forces the model to 
learn an "average" response that applies to no store particularly well.

**Alternative Strategy — Store Cluster + Segment Models:**

1. **Cluster stores** into groups using K-Means on attributes like 
   `location_type`, `competition_density`, `average_footfall`, and 
   `store_size`.
2. **Train one model per cluster** — each model learns patterns specific 
   to stores with similar characteristics.
3. **Optionally: Hierarchical / Mixed-Effects model** — a global model 
   with store-specific intercepts (random effects), allowing shared 
   learning while respecting store-level variation.

This approach respects heterogeneity across stores while avoiding the 
data sparsity problem of training 50 completely separate models.

---

## B2. Data and EDA Strategy

### B2(a) — Joining the Four Tables (4 marks)

**The four tables:**
- `transactions`: One row per transaction — includes `transaction_id`, 
  `store_id`, `transaction_date`, `items_sold`
- `store_attributes`: One row per store — `store_id`, `store_size`, 
  `location_type`, `monthly_footfall`, `competition_density`
- `promotion_details`: One row per promotion-month-store assignment — 
  `store_id`, `month`, `promotion_type`
- `calendar`: One row per date — `date`, `is_weekend`, `is_festival`

**Join sequence:**
1. Aggregate `transactions` to the **store × month** grain:  
   `GROUP BY store_id, YEAR(date), MONTH(date)` → `SUM(items_sold)`
2. Join `promotion_details` on `store_id` + `month/year`
3. Join `store_attributes` on `store_id`
4. Join `calendar` on `month/year` (aggregate `is_festival` as max, 
   `is_weekend` as count of weekend days)

**Grain of final modelling dataset:**  
**One row = one store × one calendar month**

**Aggregations before modelling:**
- `items_sold`: SUM within the month per store
- `is_festival`: MAX (1 if any festival day occurred that month)
- Weekend days: COUNT of weekend days in that month (or flag the month)

---

### B2(b) — EDA Strategy: 4 Key Analyses (4 marks)

**1. Promotion effectiveness by type (Bar chart: mean items_sold per promotion_type)**  
→ Reveals which promotions drive the most volume on average across all stores.  
→ Influences: If one promotion dominates, we need to investigate whether 
it's truly effective or just used in stronger stores (confounding).

**2. Location × Promotion interaction heatmap (pivot table: mean items_sold)**  
→ Shows whether promotion effectiveness varies by location type.  
→ Influences: If BOGO works only in urban stores, we'd engineer an 
interaction feature: `promotion_type × location_type`.

**3. Time-series plot of monthly items_sold per store cluster**  
→ Reveals seasonal patterns (e.g., December peaks), trends (growth/decline), 
and any anomalies (lockdowns, store closures).  
→ Influences: Whether to include month, quarter, and year features, 
and whether a temporal split is needed (it is).

**4. Distribution of items_sold with/without any promotion (Box plot)**  
→ Directly tests the hypothesis that promotions increase sales volume.  
→ Influences: Whether to include a binary `has_promotion` feature.

---

### B2(c) — Handling 80% Non-Promotion Records (2 marks)

**The imbalance problem:**  
If 80% of transactions have no promotion, the model can learn to predict 
"promotion doesn't matter much" simply by fitting the majority pattern — 
leading to underestimation of promotion effects.

**Steps to address it:**

1. **Stratified sampling:** Ensure both promotion and no-promotion records 
   are represented proportionally in train/test splits.

2. **Separate modelling:** Train one model to predict baseline sales 
   (no promotion) and a second to predict the **uplift** caused by each 
   promotion type. The final recommendation = baseline + maximum uplift.

3. **Oversampling promotion records** using techniques like SMOTE 
   (for classification variants) or weighted regression (assign higher 
   loss weight to promotion observations).

4. **Feature engineering:** Create a binary `has_promotion` column and 
   interaction terms to help the model explicitly distinguish between 
   the two regimes.

---

## B3. Model Evaluation and Deployment

### B3(a) — Train-Test Split and Metrics (4 marks)

**Setup for 3-year, 50-store monthly data:**  
Total data points: ~50 stores × 36 months = ~1,800 rows.

**Temporal split strategy:**  
- Training set: Months 1–30 (first 2.5 years)  
- Test set: Months 31–36 (last 6 months)

This ensures the model is evaluated on genuinely unseen future data, 
simulating the real deployment scenario.

**Why random split is inappropriate:**  
Random splitting leaks future information into training (e.g., a December 
2024 row in training while March 2024 is in testing), inflating performance 
metrics unrealistically. In production, you can only know the past — never 
randomly sampled future months.

**Evaluation metrics:**

| Metric | Formula | Business Interpretation |
|--------|---------|------------------------|
| **RMSE** | √mean((actual − predicted)²) | Penalises large errors. Useful when over-stocking 500 items is worse than over-stocking 50 items. |
| **MAE** | mean(|actual − predicted|) | Average absolute error in items. Easy to explain: "On average, our predictions are off by X items." |
| **MAPE** | mean(|actual − predicted| / actual) × 100 | Percentage error — useful for comparing stores of different sizes. |
| **Per-promotion RMSE** | RMSE computed per promotion type | Shows which promotions are hardest to predict — actionable for targeted improvement. |

---

### B3(b) — Explaining Different Recommendations via Feature Importance (4 marks)

**Why Store 12 gets different recommendations in December vs March:**

The model makes recommendations by predicting `items_sold` for each of 
the 5 promotion types for Store 12 in that month and recommending the one 
with the highest predicted value.

**How to investigate using feature importance:**

1. **Global feature importance** (from Random Forest): Identify which 
   features have the highest importance scores overall. If `month` ranks 
   highly, it means seasonal timing strongly drives predictions.

2. **SHAP values (SHapley Additive exPlanations):** For a single prediction 
   (Store 12, December, Loyalty Points), SHAP shows how each feature 
   *pushed the prediction* up or down from the average. You'd see that:
   - `month=12` contributes a large positive effect (December peak)
   - `is_festival=1` further boosts the Loyalty Points recommendation
   - `competition_density` contributes less

3. **Explanation to marketing team:**  
   "In December, the model predicts Loyalty Points Bonus will drive the 
   most sales for Store 12 because December is a high-footfall month 
   (seasonal effect) and customers respond well to loyalty incentives 
   during gift-giving season. In March, there are no festivals, footfall 
   is lower, and Flat Discounts are predicted to better stimulate 
   demand among cost-conscious shoppers post-holiday period."

This narrative combines what the model found with business intuition — 
making it actionable for a non-technical audience.

---

### B3 (c) — End-to-End Deployment Pipeline (4 marks)

**Step 1 — Save the trained model:**
```python
import joblib
joblib.dump(pipeline_rf, 'promotion_recommendation_model.pkl')
```
Save the complete pipeline (preprocessor + model) so preprocessing 
is automatically applied to new data in production.

**Step 2 — Monthly data preparation:**
At the start of each month, the data team runs a script that:
1. Pulls the latest store attributes from the database
2. Appends the current month's calendar flags (festivals, weekends)
3. Generates 5 rows per store — one per promotion type
4. Feeds this into the saved pipeline to predict `items_sold` per option
5. The promotion with the highest predicted value is recommended per store

**Step 3 — Generating recommendations:**
```python
model = joblib.load('promotion_recommendation_model.pkl')
new_data = prepare_monthly_features(stores_df, current_month)
predictions = model.predict(new_data)
# Select top promotion per store based on predictions
```

**Step 4 — Monitoring (detecting model degradation):**

| Signal | How to detect | Action |
|--------|--------------|--------|
| **Performance drift** | Compare actual vs predicted MAE monthly | Trigger retraining if MAE exceeds threshold |
| **Data drift** | Monitor distribution of input features (e.g., footfall dropped post-COVID) | Retrain with recent data |
| **Concept drift** | Customer behaviour shifts (new competitors, economic changes) | Add recent data, potentially add new features |
| **Prediction distribution** | Track if model always recommends same promotion | Investigate feature drift |

**Retraining trigger:**  
Use a rolling window strategy — retrain every quarter using the last 
12 months of data. Automated alerts fire when test-month MAE rises 
more than 15% above the baseline.