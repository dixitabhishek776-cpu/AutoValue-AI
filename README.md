# AutoValue AI — Used Car Price Prediction

A Streamlit machine-learning prototype for estimating used-car resale prices in INR.

## Important data disclosure
The bundled `data/sample/demo_used_cars.csv` is **synthetic demonstration data**.
It is not scraped or sourced from CarDekho and must not be represented as real market data.
Model metrics produced from it are only pipeline smoke-test metrics, not evidence of real-world accuracy.

## Run locally (Termux / Linux / macOS)
```bash
pkg install python git -y  # Termux only
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py --server.address 0.0.0.0
```
Open `http://127.0.0.1:8501`.

## Use a real dataset
Prepare a CSV with these exact columns:
`brand,year,km_driven,fuel,transmission,owner,mileage,engine_cc,seats,selling_price`

Then run:
```bash
python train.py --data path/to/your_licensed_dataset.csv
```
Check column meanings, currency, units, duplicate listings, missing values, target definition,
date coverage and licensing before training or publishing results.

## Evaluation
Training creates `reports/model_metrics.json` with MAE, RMSE and R² on a held-out split.
Do not compare metrics across datasets with different target definitions or sampling methods.

## Project modules
- `app.py`: Streamlit user interface
- `train.py`: preprocessing, training and evaluation pipeline
- `src/schema.py`: schema validation
- `src/valuation.py`: inference helpers
- `src/analytics.py`: dataset analytics
- `tests/`: basic schema tests

## Limitations
This is a project prototype, not a certified vehicle appraisal service. It does not verify accident
history, service records, ownership documents, local demand, or live listings.
