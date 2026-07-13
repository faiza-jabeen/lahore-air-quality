"""Generate a realistic synthetic Lahore air quality dataset with intentional
messiness (missing values, duplicates, inconsistent labels) so the cleaning
pipeline has something real to do."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
stations = ["Gulberg", "Township", "DHA", "Johar Town", "Walled City"]

rows = []
for station in stations:
    # station-level baseline: Walled City dirtiest, DHA cleanest
    base = {"Gulberg": 95, "Township": 110, "DHA": 78,
            "Johar Town": 92, "Walled City": 125}[station]

    for d in dates:
        doy = d.dayofyear
        # strong winter peak (smog season), summer trough
        seasonal = 55 * np.cos(2 * np.pi * (doy - 15) / 365)
        # weekly: slightly lower on Sunday
        weekly = -8 if d.dayofweek == 6 else 0
        # slow upward trend year over year
        trend = 0.02 * (d - dates[0]).days

        temp = 24 + 12 * np.sin(2 * np.pi * (doy - 100) / 365) + rng.normal(0, 3)
        humidity = 55 + 20 * np.sin(2 * np.pi * (doy - 200) / 365) + rng.normal(0, 8)
        wind = np.abs(rng.gamma(2.0, 1.6))
        rain = max(0.0, rng.gamma(0.35, 6.0) - 1.2) if 150 < doy < 260 else max(0.0, rng.gamma(0.15, 3.0) - 1.0)

        # wind and rain clear the air
        pm25 = (base + seasonal + weekly + trend
                - 4.5 * wind - 1.8 * rain
                + 0.35 * (humidity - 55)
                + rng.normal(0, 12))
        pm25 = float(np.clip(pm25, 8, 400))

        rows.append({
            "date": d.strftime("%Y-%m-%d"),
            "station": station,
            "pm25": round(pm25, 1),
            "temperature_c": round(temp, 1),
            "humidity_pct": round(np.clip(humidity, 10, 100), 1),
            "wind_speed_kmh": round(wind, 2),
            "rainfall_mm": round(rain, 1),
        })

df = pd.DataFrame(rows)

# --- inject realistic messiness ---
# 1. missing values scattered through sensor columns
for col, frac in [("pm25", 0.04), ("humidity_pct", 0.03), ("wind_speed_kmh", 0.02)]:
    idx = rng.choice(df.index, size=int(len(df) * frac), replace=False)
    df.loc[idx, col] = np.nan

# 2. inconsistent station labels (whitespace / casing)
idx = rng.choice(df.index, size=120, replace=False)
df.loc[idx, "station"] = df.loc[idx, "station"].str.upper()
idx = rng.choice(df.index, size=90, replace=False)
df.loc[idx, "station"] = " " + df.loc[idx, "station"].astype(str) + " "

# 3. duplicate rows
dupes = df.sample(60, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)

# 4. impossible sentinel values (common in real sensor data)
idx = rng.choice(df.index, size=40, replace=False)
df.loc[idx, "pm25"] = -999

# 5. shuffle so it doesn't look pre-sorted
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

df.to_csv("/home/claude/project/data/lahore_air_quality_raw.csv", index=False)
print(f"rows: {len(df)}, cols: {list(df.columns)}")
print(df.head(3).to_string())
