# AS-TestDataScience-2

Multivariate time series forecasting of home energy consumption using the **Appliances Energy Prediction** dataset (UCI).  
This project aims to predict future appliance energy use without access to future values of the regressors.  
It includes preprocessing, stationarity checks, model training, evaluation, and reproducibility features.

---

## 📂 Project Structure

This repository follows the [cookiecutter data science](https://drivendata.github.io/cookiecutter-data-science/) structure:

```
├── LICENSE             <- MIT License.
├── Makefile            <- Reproducible commands: make data, make train, etc.
├── README.md           <- Project overview and instructions.
├── models/             <- Trained models (.pth) and their training curves.
├── notebooks/          <- Jupyter notebooks for analysis and training.
├── reports/            <- Final reports and figures.
├── requirements.txt    <- Python dependencies.
├── setup.py            <- Installation script (`pip install -e .`)
├── src/                <- All source code (data prep, features, models, viz).
└── tox.ini             <- Linting rules.
```

---

## 📄 Documentation

This MkDocs site provides structured technical documentation.  
Additionally, the following resources are available for direct exploration:

- ✅ **Executed Jupyter notebook** (interactive report): [`reports/energy_prediction.html`](as-testdatascience-2/reports/energy_prediction.html)  
- 🎞 **Presentation slides** (PDF format): [`reports/as_testdatascience_2.pdf`](as-testdatascience-2/reports/as_testdatascience_2.pdf)

These files allow for quick visualization of results and core methods without running code manually.

---

## 📊 Dataset

**Appliances Energy Prediction** ([UCI Repository](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction)):

- **Time range:** ~4.5 months, sampled every 10 minutes (19,735 entries)
- **Features:** 28 sensor variables (temperature, humidity, weather, random)
- **Target:** Appliance energy use (`Appliances`, in Wh)
- **No missing values**, rich periodicity (daily and sub-daily)

---

## 🧹 Pipeline Overview

1. **Data Loading & Cleaning**
   - Standardize datetime format
   - Set time index
   - Normalize features using `StandardScaler` (fit on training set only)

2. **EDA & Stationarity**
   - Rolling statistics, ACF, FFT, spectrogram
   - ADF tests (global + rolling)
   - Seasonal decomposition

3. **Supervised Learning Setup**
   - Sliding window with:
     - Input length = 5 days
     - Horizon = 100 steps
     - Stride = 12 hours
   - Custom PyTorch dataset and dataloaders

4. **Model Training**
   - Models: LSTM, Transformer, TCN
   - Loss: MAE
   - Optimizer: Adam
   - Trained for 200 epochs

5. **Evaluation**
   - Metrics: MAE, RMSE
   - Visual comparison of predictions
   - Model checkpoints and loss curves saved in `models/`

---

## 📈 Model Comparison

| Model        | MAE   | RMSE  |
|--------------|-------|-------|
| **LSTM**     | 0.570 | 1.105 |
| **Transformer** | **0.532** | **1.062** |
| **TCN**      | 0.607 | 1.084 |

The Transformer achieves the best performance overall, followed closely by LSTM and TCN.

---

## 💡 Key Takeaways

- **Transformer-based models** are highly effective for this type of multivariate sequence regression.
- **Daily seasonality** is dominant and must be considered during modeling and validation.
- Despite the global ADF test suggesting stationarity, **rolling ADF and visual checks reveal clear non-stationarity**.
- This project demonstrates a **reproducible deep learning forecasting pipeline** suitable for real-world deployment and extensibility.

---

## 🔧 Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/antosquicciarini/as-testdatascience-2.git
   cd as-testdatascience-2
   ```

2. **Create and activate environment** (automated via `make`):
   ```bash
   make create_environment
   ```
   If conda is available, then 
   ```bash
   conda activate as-testdatascience-2
   ```
3. **Install dependencies**:
   ```bash
   make requirements
   ```

4. **Validate environment setup**:
   ```bash
   make test_environment
   ```

5. **Run training**:
   run notebooks/energy_prediction.ipynb via Jupyter

---

## 🧪 Testing Your Environment

Run:

```bash
python test_environment.py
```

to confirm your Python version is compatible.

---

## 🧪 Testing & Code Quality

To ensure the code follows Python PEP8 style guidelines, you can run:

```bash
make lint
```
---

## ⚙️ DevOps Readiness

- Project structured with automation via Makefile
- Environment reproducibility via requirements.txt and setup.py
- Linting, testing, and documentation support (tox, flake8, Sphinx)
- Models saved and ready for deployment in production systems

---

## 📜 License

This project is licensed under the **MIT License**. See `LICENSE` for full details.

---

<p><small>Project scaffolded with <a href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science</a>.</small></p>