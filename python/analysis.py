"""
Lahore Air Quality: End-to-End Analysis
=======================================
Cleans raw sensor data, explores seasonal and spatial patterns in PM2.5,
and builds models to identify what actually drives air pollution in Lahore.

Author: Faiza Jabeen
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sns.set_theme(style="whitegrid")
PALETTE = "rocket"
OUT = "outputs"

# WHO 24-hour guideline for PM2.5
WHO_GUIDELINE = 15


# ----------------------------------------------------------------------
# 1. LOAD & CLEAN
# ----------------------------------------------------------------------
def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw sensor data and fix the problems real datasets always have."""
    df = pd.read_csv(path)
    print(f"Raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # Station names arrive with inconsistent casing and stray whitespace
    df["station"] = df["station"].str.strip().str.title()

    # -999 is a sentinel for 'sensor failed', not a real reading
    df["pm25"] = df["pm25"].replace(-999, np.nan)

    # Exact duplicate rows are logging errors
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")

    df["date"] = pd.to_datetime(df["date"])

    # Report missingness before deciding what to do about it
    missing = df.isna().sum()
    print("\nMissing values:")
    print(missing[missing > 0].to_string())

    # PM2.5 is the target: rows without it can't be used for modelling
    df = df.dropna(subset=["pm25"])

    # Weather gaps are filled per-station by interpolating over time,
    # which respects the fact that weather is autocorrelated day to day
    df = df.sort_values(["station", "date"])
    for col in ["humidity_pct", "wind_speed_kmh", "temperature_c"]:
        df[col] = df.groupby("station")[col].transform(
            lambda s: s.interpolate().bfill().ffill()
        )

    print(f"\nClean data: {df.shape[0]} rows")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the time and lag features that make the models meaningful."""
    df = df.copy()
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # Smog season in Lahore runs roughly November to February
    df["is_smog_season"] = df["month"].isin([11, 12, 1, 2]).astype(int)

    # Cyclical encoding: day 365 and day 1 are neighbours, not opposites
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

    # Yesterday's pollution is the strongest single predictor of today's
    df = df.sort_values(["station", "date"])
    df["pm25_lag1"] = df.groupby("station")["pm25"].shift(1)
    df["pm25_roll7"] = (
        df.groupby("station")["pm25"].shift(1).rolling(7, min_periods=1).mean()
    )

    return df.dropna(subset=["pm25_lag1"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# 3. EXPLORATORY ANALYSIS
# ----------------------------------------------------------------------
def explore(df: pd.DataFrame) -> None:
    """Produce the figures that tell the story."""

    # --- Figure 1: seasonal cycle, the headline finding ---
    fig, ax = plt.subplots(figsize=(11, 5))
    monthly = df.groupby("month")["pm25"].agg(["mean", "std"])
    ax.plot(monthly.index, monthly["mean"], marker="o", lw=2.5, color="#B23A48")
    ax.fill_between(
        monthly.index,
        monthly["mean"] - monthly["std"],
        monthly["mean"] + monthly["std"],
        alpha=0.18, color="#B23A48",
    )
    ax.axhline(WHO_GUIDELINE, ls="--", color="#2A9D8F", lw=1.8,
               label=f"WHO guideline ({WHO_GUIDELINE} µg/m³)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Lahore PM2.5 by month: smog season is a different world",
                 fontsize=13, weight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/01_seasonal_cycle.png", dpi=150)
    plt.close()

    # --- Figure 2: which neighbourhoods bear the burden ---
    fig, ax = plt.subplots(figsize=(9, 5))
    order = df.groupby("station")["pm25"].median().sort_values().index
    sns.boxplot(data=df, x="station", y="pm25", order=order,
                hue="station", palette=PALETTE, legend=False, ax=ax)
    ax.axhline(WHO_GUIDELINE, ls="--", color="#2A9D8F", lw=1.8)
    ax.set_xlabel("")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("Pollution is not shared equally across Lahore",
                 fontsize=13, weight="bold")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{OUT}/02_station_comparison.png", dpi=150)
    plt.close()

    # --- Figure 3: what weather does to the air ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, (col, label) in zip(axes, [
        ("wind_speed_kmh", "Wind speed (km/h)"),
        ("humidity_pct", "Humidity (%)"),
        ("temperature_c", "Temperature (°C)"),
    ]):
        ax.scatter(df[col], df["pm25"], s=6, alpha=0.25, color="#4A5859")
        # LOWESS-free trend: bin and take the mean, easy to read and defend
        bins = pd.qcut(df[col], 12, duplicates="drop")
        trend = df.groupby(bins, observed=True).agg(
            x=(col, "mean"), y=("pm25", "mean")
        )
        ax.plot(trend["x"], trend["y"], color="#B23A48", lw=2.5)
        ax.set_xlabel(label)
        ax.set_ylabel("PM2.5 (µg/m³)" if col == "wind_speed_kmh" else "")
    fig.suptitle("Wind clears the air; humidity traps it",
                 fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/03_weather_relationships.png", dpi=150)
    plt.close()

    # --- Figure 4: correlations ---
    fig, ax = plt.subplots(figsize=(7.5, 6))
    cols = ["pm25", "temperature_c", "humidity_pct",
            "wind_speed_kmh", "rainfall_mm", "pm25_lag1"]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlations", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/04_correlations.png", dpi=150)
    plt.close()

    print(f"\nFigures written to {OUT}/")


# ----------------------------------------------------------------------
# 4. MODELLING
# ----------------------------------------------------------------------
def model(df: pd.DataFrame) -> pd.DataFrame:
    """Compare a simple baseline against a flexible model.

    Validation uses TimeSeriesSplit, not random k-fold. Random splits would
    let the model peek at future days to predict past ones, which inflates
    scores and would never survive contact with real deployment.
    """
    features = [
        "temperature_c", "humidity_pct", "wind_speed_kmh", "rainfall_mm",
        "doy_sin", "doy_cos", "is_weekend", "is_smog_season",
        "pm25_lag1", "pm25_roll7",
    ]

    df = df.sort_values("date").reset_index(drop=True)
    X, y = df[features], df["pm25"]

    cv = TimeSeriesSplit(n_splits=5)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
    }

    results = []
    for name, mdl in models.items():
        mae = -cross_val_score(mdl, X, y, cv=cv,
                               scoring="neg_mean_absolute_error").mean()
        r2 = cross_val_score(mdl, X, y, cv=cv, scoring="r2").mean()
        results.append({"model": name, "MAE": round(mae, 2), "R2": round(r2, 3)})
        print(f"{name:20s}  MAE = {mae:6.2f} µg/m³   R² = {r2:.3f}")

    # Feature importance from the fitted Random Forest
    rf = models["Random Forest"].fit(X, y)
    imp = (pd.Series(rf.feature_importances_, index=features)
             .sort_values(ascending=True))

    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot.barh(ax=ax, color="#7A6C5D")
    ax.set_xlabel("Importance")
    ax.set_title("What actually drives PM2.5?", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT}/05_feature_importance.png", dpi=150)
    plt.close()

    print("\nTop drivers:")
    print(imp.sort_values(ascending=False).head(4).to_string())

    return pd.DataFrame(results)


# ----------------------------------------------------------------------
def main():
    print("=" * 62)
    print("LAHORE AIR QUALITY ANALYSIS")
    print("=" * 62)

    df = load_and_clean("data/lahore_air_quality_raw.csv")
    df = add_features(df)
    df.to_csv("data/lahore_air_quality_clean.csv", index=False)

    print("\n" + "-" * 62)
    print("KEY NUMBERS")
    print("-" * 62)
    mean_pm = df["pm25"].mean()
    smog = df[df["is_smog_season"] == 1]["pm25"].mean()
    clear = df[df["is_smog_season"] == 0]["pm25"].mean()
    exceed = (df["pm25"] > WHO_GUIDELINE).mean() * 100

    print(f"Mean PM2.5                : {mean_pm:.1f} µg/m³")
    print(f"Smog season (Nov-Feb)     : {smog:.1f} µg/m³")
    print(f"Rest of year              : {clear:.1f} µg/m³")
    print(f"Ratio                     : {smog/clear:.2f}x worse")
    print(f"Days above WHO guideline  : {exceed:.1f}%")
    print(f"Multiple of WHO guideline : {mean_pm/WHO_GUIDELINE:.1f}x")

    print("\n" + "-" * 62)
    print("EXPLORATORY ANALYSIS")
    print("-" * 62)
    explore(df)

    print("\n" + "-" * 62)
    print("MODELLING (TimeSeriesSplit CV)")
    print("-" * 62)
    results = model(df)
    results.to_csv(f"{OUT}/model_results.csv", index=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
