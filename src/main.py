"""
FDA Car Price Prediction
Author: Arpit Parashar

How to control the output:

1. RAW mode:
   VISUALISATION_MODE = "raw"
   This shows only raw/unclean data visualisations.
   It does not clean the dataframe.

2. CLEAN mode:
   VISUALISATION_MODE = "clean"
   This cleans the data first, then shows cleaned visualisations and models.

3. BOTH mode:
   VISUALISATION_MODE = "both"
   This creates raw visualisations first, then cleaned visualisations and models.

Only change the two settings below.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# SIMPLE SETTINGS - CHANGE ONLY THIS PART

# Choose one: "raw", "clean", or "both"
VISUALISATION_MODE = "both"

# Keep True if you want regression + classification results in clean/both mode.
RUN_MODELS = True

# PROJECT PATHS

# Works whether this file is saved as project_root/run.py or project_root/src/run.py
CURRENT_FILE_DIR = Path(__file__).resolve().parent

if (CURRENT_FILE_DIR / "data" / "auto.csv").exists():
    BASE_DIR = CURRENT_FILE_DIR
else:
    BASE_DIR = CURRENT_FILE_DIR.parent

DATA_PATH = BASE_DIR / "data" / "auto.csv"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"
TABLE_DIR = BASE_DIR / "outputs" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration",
    "num-of-doors", "body-style", "drive-wheels", "engine-location",
    "wheel-base", "length", "width", "height", "curb-weight",
    "engine-type", "num-of-cylinders", "engine-size", "fuel-system",
    "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm",
    "city-mpg", "highway-mpg", "price"
]


# BASIC HELPER FUNCTIONS

def save_figure(file_name: str) -> None:
    """Save the current matplotlib figure."""
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / file_name, dpi=180, bbox_inches="tight")
    plt.close()


def load_raw_data() -> pd.DataFrame:
    """Load the original dataset and assign column headers."""
    df_raw = pd.read_csv(DATA_PATH, header=None, names=HEADERS)
    return df_raw


def save_basic_raw_outputs(df_raw: pd.DataFrame) -> None:
    """Save basic raw EDA tables before cleaning."""
    df_raw.head(10).to_csv(TABLE_DIR / "raw_first_10_rows.csv", index=False)

    df_raw.dtypes.astype(str).reset_index().rename(
        columns={"index": "Column", 0: "Raw data type"}
    ).to_csv(TABLE_DIR / "raw_data_types.csv", index=False)

    raw_stats = df_raw.describe(include="all").transpose()
    raw_stats.to_csv(TABLE_DIR / "raw_descriptive_statistics.csv")

    categorical_cols = df_raw.select_dtypes(include="object").columns
    unique_rows = []

    for col in categorical_cols:
        values = df_raw[col].astype(str).unique().tolist()
        unique_rows.append({
            "Column": col,
            "Unique values count": len(values),
            "Sample values": ", ".join(values[:8])
        })

    pd.DataFrame(unique_rows).to_csv(
        TABLE_DIR / "raw_categorical_unique_values.csv",
        index=False
    )


def create_heatmap(corr_df: pd.DataFrame, title: str, file_name: str) -> None:
    """
    Create a readable correlation heatmap with numbers inside each cell.
    This function is used for both raw and clean heatmaps.
    """
    if corr_df.empty:
        print(f"Skipping heatmap because correlation data is empty: {title}")
        return

    plt.figure(figsize=(12, 9))
    im = plt.imshow(corr_df, aspect="auto", cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(
        range(len(corr_df.columns)),
        corr_df.columns,
        rotation=45,
        ha="right",
        fontsize=9
    )

    plt.yticks(
        range(len(corr_df.index)),
        corr_df.index,
        fontsize=9
    )

    plt.title(title, fontsize=12)

    # Add correlation values inside each cell
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            value = corr_df.iloc[i, j]

            if pd.isna(value):
                display_value = ""
                text_color = "black"
            else:
                display_value = f"{value:.2f}"
                text_color = "white" if abs(value) > 0.5 else "black"

            plt.text(
                j,
                i,
                display_value,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8
            )

    save_figure(file_name)


# RAW / UNCLEAN VISUALISATION

def create_raw_visualisations(df_raw: pd.DataFrame) -> None:
    """
    Create raw data visualisations without cleaning the dataframe.

    Important:
    - No '?' replacement
    - No NaN replacement
    - No mean/mode replacement
    - No type conversion
    - No outlier treatment
    - No normalisation
    - No binning

    These graphs intentionally look messy because the raw data still contains
    text values and '?' markers.
    """
    print("\nRAW DATA VISUALISATION MODE")
    print("No cleaning has been applied.")
    print("Raw dataset shape:", df_raw.shape)
    print("\nFirst 10 rows of raw data:")
    print(df_raw.head(10).to_string(index=False))

    save_basic_raw_outputs(df_raw)

    # Raw price values as text/counts
    plt.figure(figsize=(12, 5))
    df_raw["price"].astype(str).value_counts(dropna=False).sort_index().plot(kind="bar")
    plt.title("Raw Price Values Before Cleaning")
    plt.xlabel("Raw price value as text")
    plt.ylabel("Count")
    plt.xticks(rotation=90, fontsize=5)
    save_figure("01_raw_price_values_before_cleaning.png")

    # Raw horsepower values as text/counts
    plt.figure(figsize=(12, 5))
    df_raw["horsepower"].astype(str).value_counts(dropna=False).sort_index().plot(kind="bar")
    plt.title("Raw Horsepower Values Before Cleaning")
    plt.xlabel("Raw horsepower value as text")
    plt.ylabel("Count")
    plt.xticks(rotation=90, fontsize=5)
    save_figure("02_raw_horsepower_values_before_cleaning.png")

    # Raw price against body-style using raw text values
    plt.figure(figsize=(12, 6))
    plt.scatter(
        df_raw["body-style"].astype(str),
        df_raw["price"].astype(str),
        alpha=0.65,
        s=18
    )
    plt.title("Raw Price vs Body Style Before Cleaning")
    plt.xlabel("Body style")
    plt.ylabel("Raw price value as text")
    plt.xticks(rotation=45)
    plt.yticks(fontsize=5)
    save_figure("03_raw_price_vs_body_style_before_cleaning.png")

    # Raw correlation heatmap
    # Only columns already numeric in the raw file will appear here.
    # This helps show why cleaning is needed.
    raw_corr = df_raw.corr(numeric_only=True)

    create_heatmap(
        corr_df=raw_corr,
        title="Raw Correlation Heatmap Before Cleaning",
        file_name="04_raw_correlation_heatmap_before_cleaning.png"
    )

    print("\nRaw visualisations saved in:", FIGURE_DIR)


# CLEANING AND FEATURE ENGINEERING

def clean_and_prepare_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean data and prepare features for visualisation and modelling."""
    df = df_raw.copy()

    # Step 1: Count '?' markers before replacing them
    question_mark_counts = (df == "?").sum()
    question_mark_counts[question_mark_counts > 0].reset_index().rename(
        columns={"index": "Column", 0: "Count of ? markers"}
    ).to_csv(TABLE_DIR / "question_mark_counts.csv", index=False)

    # Step 2: Replace '?' with NaN
    df.replace("?", np.nan, inplace=True)

    missing_counts = df.isna().sum()
    missing_counts[missing_counts > 0].reset_index().rename(
        columns={"index": "Column", 0: "Missing values"}
    ).to_csv(TABLE_DIR / "missing_values_after_replacing_question_mark.csv", index=False)

    # Step 3: Convert numeric-like columns to numeric
    numeric_like_columns = [
        "normalized-losses", "bore", "stroke",
        "horsepower", "peak-rpm", "price"
    ]

    for col in numeric_like_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Step 4: Mean replacement for numeric columns
    mean_rows = []

    for col in df.select_dtypes(include="number").columns:
        if df[col].isna().any():
            mean_value = df[col].mean()
            df[col] = df[col].fillna(mean_value)
            mean_rows.append({
                "Column": col,
                "Mean used for replacement": round(mean_value, 3)
            })

    pd.DataFrame(mean_rows).to_csv(
        TABLE_DIR / "mean_replacements.csv",
        index=False
    )

    # Step 5: Mode replacement for categorical columns
    mode_rows = []

    for col in df.select_dtypes(include="object").columns:
        if df[col].isna().any():
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)
            mode_rows.append({
                "Column": col,
                "Mode used for replacement": mode_value
            })

    pd.DataFrame(mode_rows).to_csv(
        TABLE_DIR / "mode_replacements.csv",
        index=False
    )

    # Step 6: Outlier capping using IQR
    outlier_rows = []

    for col in ["price", "horsepower", "engine-size", "curb-weight"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_cap = q1 - 1.5 * iqr
        upper_cap = q3 + 1.5 * iqr

        outlier_count = ((df[col] < lower_cap) | (df[col] > upper_cap)).sum()

        outlier_rows.append({
            "Column": col,
            "Q1": round(q1, 3),
            "Q3": round(q3, 3),
            "IQR": round(iqr, 3),
            "Lower cap": round(lower_cap, 3),
            "Upper cap": round(upper_cap, 3),
            "Outlier count": int(outlier_count)
        })

        df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)

    pd.DataFrame(outlier_rows).to_csv(
        TABLE_DIR / "outlier_summary.csv",
        index=False
    )

    # Step 7: Feature engineering
    df["city-L/100km"] = 235 / df["city-mpg"]

    # Step 8: Binning
    df["horsepower-binned"] = pd.cut(
        df["horsepower"],
        bins=3,
        labels=["Low", "Medium", "High"]
    )

    # Step 9: Normalisation
    df["length-normalized"] = df["length"] / df["length"].max()
    df["width-normalized"] = df["width"] / df["width"].max()

    clean_stats = df[[
        "price", "horsepower", "engine-size", "curb-weight",
        "city-mpg", "highway-mpg", "city-L/100km",
        "length-normalized", "width-normalized"
    ]].agg(["mean", "median", "std"]).transpose()

    clean_stats.to_csv(TABLE_DIR / "clean_descriptive_statistics.csv")

    return df


# CLEAN VISUALISATION

def create_clean_visualisations(df: pd.DataFrame) -> None:
    """Create visualisations after cleaning and preprocessing."""
    print("\nCLEAN DATA VISUALISATION MODE")
    print("Data cleaning, outlier treatment, feature engineering, binning and normalisation are applied.")
    print("\nCleaned descriptive statistics:")
    print(df[["price", "horsepower", "engine-size", "curb-weight", "city-L/100km"]].describe().round(3))

    # Clean price histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df["price"], bins=20, edgecolor="black")
    plt.title("Cleaned Price Distribution")
    plt.xlabel("Price")
    plt.ylabel("Frequency")
    save_figure("05_cleaned_price_distribution.png")

    # Clean horsepower histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df["horsepower"], bins=20, edgecolor="black")
    plt.title("Cleaned Horsepower Distribution")
    plt.xlabel("Horsepower")
    plt.ylabel("Frequency")
    save_figure("06_cleaned_horsepower_distribution.png")

    # Clean boxplot: price vs body-style
    plt.figure(figsize=(9, 5))
    body_styles = df["body-style"].unique().tolist()
    values = [df.loc[df["body-style"] == style, "price"] for style in body_styles]

    # Matplotlib-version-friendly method:
    # Do not use plt.boxplot(values, labels=body_styles)
    plt.boxplot(values)
    plt.xticks(range(1, len(body_styles) + 1), body_styles)

    plt.title("Cleaned Price vs Body Style")
    plt.xlabel("Body style")
    plt.ylabel("Price")
    plt.xticks(rotation=25)
    save_figure("07_cleaned_price_vs_body_style_boxplot.png")

    # Clean correlation heatmap using the reusable function
    corr_cols = [
        "price", "horsepower", "engine-size", "curb-weight",
        "city-mpg", "highway-mpg", "city-L/100km", "length", "width"
    ]

    clean_corr = df[corr_cols].corr()

    create_heatmap(
        corr_df=clean_corr,
        title="Cleaned Correlation Heatmap",
        file_name="08_cleaned_correlation_heatmap.png"
    )

    # Horsepower bin chart
    plt.figure(figsize=(7, 5))
    df["horsepower-binned"].value_counts().sort_index().plot(kind="bar")
    plt.title("Horsepower Binning: Low, Medium and High")
    plt.xlabel("Horsepower category")
    plt.ylabel("Number of cars")
    plt.xticks(rotation=0)
    save_figure("09_horsepower_binning.png")

    # Normalised length distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["length-normalized"], bins=20, edgecolor="black")
    plt.title("Normalised Length Distribution")
    plt.xlabel("Normalised length")
    plt.ylabel("Frequency")
    save_figure("10_normalised_length_distribution.png")

    print("\nClean visualisations saved in:", FIGURE_DIR)


# MODELLING

def run_regression_and_classification(df: pd.DataFrame) -> None:
    """Run regression and classification models after cleaning."""
    print("\nMODEL DEVELOPMENT MODE")

    features = [
        "horsepower", "engine-size", "curb-weight",
        "highway-mpg", "city-L/100km", "width", "length"
    ]

    X = df[features]
    y = df["price"]

    regression_rows = []

    for split_name, test_size in [("80/20", 0.20), ("70/30", 0.30)]:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=200,
                random_state=42
            )
        }

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            regression_rows.append({
                "Split": split_name,
                "Model": model_name,
                "R2": round(r2_score(y_test, predictions), 3),
                "MAE": round(mean_absolute_error(y_test, predictions), 3),
                "MSE": round(mean_squared_error(y_test, predictions), 3),
                "RMSE": round(np.sqrt(mean_squared_error(y_test, predictions)), 3)
            })

            # Actual vs predicted plot
            if split_name == "80/20":
                plt.figure(figsize=(7, 5))
                plt.scatter(y_test, predictions, alpha=0.75)
                plt.xlabel("Actual price")
                plt.ylabel("Predicted price")
                plt.title(f"{model_name}: Actual vs Predicted Prices")

                min_value = min(y_test.min(), predictions.min())
                max_value = max(y_test.max(), predictions.max())

                plt.plot(
                    [min_value, max_value],
                    [min_value, max_value],
                    linestyle="--"
                )

                safe_name = model_name.lower().replace(" ", "_")
                save_figure(f"11_{safe_name}_actual_vs_predicted.png")

    regression_results = pd.DataFrame(regression_rows)
    regression_results.to_csv(TABLE_DIR / "regression_metrics.csv", index=False)

    print("\nRegression metrics:")
    print(regression_results.to_string(index=False))

    # Cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_rows = []

    cv_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
    }

    for model_name, model in cv_models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="r2")

        cv_rows.append({
            "Model": model_name,
            "CV mean R2": round(scores.mean(), 3),
            "CV std R2": round(scores.std(), 3),
            "Fold scores": "; ".join([str(round(score, 3)) for score in scores])
        })

    cv_results = pd.DataFrame(cv_rows)
    cv_results.to_csv(TABLE_DIR / "cross_validation_results.csv", index=False)

    print("\nCross-validation results:")
    print(cv_results.to_string(index=False))

    # Classification
    # TASK 4: Convert price into Low, Medium and High categories
    labels = ["Low", "Medium", "High"]

    df["price-category"] = pd.qcut(
        df["price"],
        q=3,
        labels=labels
    )

    X_class = df[features]
    y_class = df["price-category"]

    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
        X_class,
        y_class,
        test_size=0.20,
        random_state=42,
        stratify=y_class
    )

    classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    classifier.fit(X_train_class, y_train_class)
    y_pred_class = classifier.predict(X_test_class)

    classification_accuracy = accuracy_score(y_test_class, y_pred_class)
    correct_predictions = int((y_test_class == y_pred_class).sum())
    test_records = int(len(y_test_class))

    print("\nTASK 4: Classification Model")
    print("Logistic Regression Accuracy:", round(classification_accuracy, 3))
    print(f"Correct classifications: {correct_predictions}/{test_records}")

    print("\nClassification Report:")
    print(classification_report(y_test_class, y_pred_class, labels=labels))

    # Table 13: Classification Performance Summary
    classification_summary = pd.DataFrame([{
        "Table": "Table 13",
        "Model": "Logistic Regression",
        "Target": "Price Category: Low / Medium / High",
        "Accuracy": round(classification_accuracy, 3),
        "Correct classifications": correct_predictions,
        "Total test records": test_records
    }])

    classification_summary.to_csv(
        TABLE_DIR / "table_13_classification_performance_summary.csv",
        index=False
    )

    # Save detailed classification report for reproducibility
    classification_report_df = pd.DataFrame(
        classification_report(
            y_test_class,
            y_pred_class,
            labels=labels,
            output_dict=True
        )
    ).transpose()

    classification_report_df.to_csv(
        TABLE_DIR / "classification_report.csv",
        index=True
    )

    print("\nTable 13: Classification Performance Summary")
    print(classification_summary.to_string(index=False))

    # Confusion matrix for classification
    cm = confusion_matrix(
        y_test_class,
        y_pred_class,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels]
    )

    cm_df.to_csv(
        TABLE_DIR / "classification_confusion_matrix.csv",
        index=True
    )

    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, aspect="auto", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.title(
        f"Logistic Regression Confusion Matrix\n"
        f"Accuracy = {classification_accuracy:.3f}"
    )
    plt.xlabel("Predicted category")
    plt.ylabel("Actual category")
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="black",
                fontsize=10
            )

    save_figure("12_classification_confusion_matrix.png")


# MAIN FUNCTION

def main() -> None:
    print("Starting FDA car price prediction project...")
    print("Dataset path:", DATA_PATH)
    print("Current VISUALISATION_MODE:", VISUALISATION_MODE)

    df_raw = load_raw_data()

    if VISUALISATION_MODE not in ["raw", "clean", "both"]:
        raise ValueError('VISUALISATION_MODE must be "raw", "clean", or "both".')

    if VISUALISATION_MODE in ["raw", "both"]:
        create_raw_visualisations(df_raw)

    if VISUALISATION_MODE in ["clean", "both"]:
        df_clean = clean_and_prepare_data(df_raw)
        create_clean_visualisations(df_clean)

        if RUN_MODELS:
            run_regression_and_classification(df_clean)
        else:
            print("\nRUN_MODELS is False, so modelling was skipped.")

    print("\nProject completed successfully.")
    print("Figures saved in:", FIGURE_DIR)
    print("Tables saved in:", TABLE_DIR)


if __name__ == "__main__":
    main()