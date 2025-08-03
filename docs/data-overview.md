# 📊 Data Overview

This project uses the **Appliances Energy Prediction Dataset** from the UCI Machine Learning Repository.  
It consists of multivariate time series measurements collected over 4.5 months in a low-energy house.  
Each instance is a 10-minute interval reading across various sensors and energy meters.

---

## 🗃 Dataset Summary

- **Observations:** 19,735 time steps  
- **Frequency:** Every 10 minutes  
- **Duration:** ~4.5 months  
- **Target:** `Appliances` (energy use in Wh)

---

## 🔍 Feature Categories

### Indoor Conditions
| Sensor | Description |
|--------|-------------|
| T1 - T9 | Temperature sensors (°C) in various rooms (e.g. kitchen, living, laundry, bathroom, bedroom, north wall) |
| RH_1 - RH_9 | Relative humidity (%) at corresponding locations |

### Weather Station Variables
| Variable | Description |
|----------|-------------|
| T_out | Outdoor temperature (°C) |
| RH_out | Outdoor humidity (%) |
| Press_mm_hg | Pressure (mm Hg) |
| Windspeed | Wind speed (m/s) |
| Visibility | Visibility (km) |
| Tdewpoint | Dew point temperature (°C) |

### Other Variables
| Variable | Description |
|----------|-------------|
| lights | Lighting energy use (Wh) |
| rv1, rv2 | Random variables with no physical interpretation |

---

## 🧪 Target Variable

- `Appliances`:  
  The primary variable to predict. Represents the energy used by appliances (in Wh).  
  It is a continuous numerical target, suitable for regression.

---

## ⚠️ Notes

- No missing values, but noise and outliers may be present.
- Features are sampled at fixed intervals; temporal order is important.
- Scaling is necessary due to different units and magnitudes.

---

📌 Source: [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)