# Phase 10 — Business Output, Excel Call List and Audit

Phase 10 converts the latest Phase 8 category scores into business-facing weekly outputs.

## Main business grain

The call list is written at:

**Customer × ZIP × Category**

Only rows successfully scored in Phase 8 are included.

The workbook is sorted by descending category Risk Score and then descending Annual Spend.

## Main workbook

File pattern:

`outputs/Weekly Call List MM-DD-YY.xlsx`

Sheets:

### Call List

Columns:

1. Customer ID
2. Customer
3. Customer Group
4. Ship-To
5. Branch
6. Sales Rep
7. Category
8. Risk Score
9. Risk Tier
10. Action
11. Annual Spend
12. Rev at Risk
13. Who They Are
14. Bold Signals
15. Grey Signals
16. Scoring Run ID

Business calculations:

```
Risk Score = round(p_14 * 100)

Annual Spend = spend_365
# trailing 365-day category spend at Customer × ZIP × Category grain

Rev at Risk = p_14 * spend_365
```

The customer×ZIP weighted-risk and account-level revenue-at-risk outputs from Phase 8 remain available as a separate rollup and are not substituted for the category-row calculations in the call list.

### Who They Are

The narrative uses only locally available pipeline fields:

- Customer Group
- tenure in years
- credit limit
- recent peak buying months
- top annual-spend categories

### Bold Signals

The strongest positive SHAP contributors (risk-up direction), up to two per scored category.

### Grey Signals

Up to three additional SHAP contributors for context.

### Column Descriptions

Business dictionary for all call-list columns.

### Lists

Hidden sheet containing approved Action values and used as the Excel Action dropdown source.

## Excel formatting

- frozen header
- filters enabled
- wrapped narrative text
- currency formatting for Annual Spend and Rev at Risk
- Risk Score conditional fills:
  - >= 40: red
  - 20–39: yellow
  - < 20: green
- Bold Signals column rendered bold
- Grey Signals column rendered grey

## Audit workbook

File pattern:

`outputs/Weekly Call List MM-DD-YY_audit.xlsx`

Sheets:

- Owner Mapping Review
- Customer ID Validation
- Validation Layer 1-3
- Row Flags
- Dormant Accounts
- Pipeline Health
- Reject Exclusions
- Customer Attributes

### Account Owner note

The production workflow resolves Account Owner from an external SharePoint mapping using Customer ID / AccountNum.

The offline synthetic replica does not fabricate this external mapping.

Therefore:

- local Sales Rep is retained as its own field
- Account Owner is left blank
- Owner Mapping Review explicitly reports:
  `OFFLINE - SharePoint owner mapping not connected`

## Validation

Output validation checks:

### Layer 1

Every Phase 8 rejected row must have a scoring ineligibility reason.

### Layer 2

- Risk Score in [0,100]
- Action belongs to approved 9-action grid
- Rev at Risk is non-negative
- Who They Are populated
- Bold Signals populated

### Layer 3

No duplicate Customer × ZIP × Category rows.

The generated workbook is reopened by the Phase 10 validator and checked for:

- exact Call List headers
- expected row count
- descending Risk Score sort
- Column Descriptions sheet
- hidden Lists sheet
- complete audit workbook sheets
- CData snapshot existence

## CData/local tabular output

A machine-readable snapshot is also generated at:

`data/output/dex_v2_cdata_output_snapshot.parquet`

This keeps the business output available for Phase 11 local analytics/dashboard work without reading Excel.

## Commands

```powershell
python -m pip install -r requirements.txt
python scripts\run_business_output.py
python scripts\validate_business_output.py
pytest -q
```

A successful output run and validator must both report:

`"status": "PASS"`
