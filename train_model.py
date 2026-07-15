import ast
import json
import re
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "house_cleaned.csv"
MODEL_PATH = BASE_DIR / "ensemble_model_best.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"

TARGET_COLUMN = "price"

BINARY_FEATURES = [
    "Private Garden / Terrace",
    "Maintenance Staff",
    "Water Storage",
    "Park",
    "Visitor Parking",
    "Waste Disposal",
    "Rain Water Harvesting",
    "Security",
]

RATING_FEATURES = [
    "Environment Rating",
    "Lifestyle Rating ",
    "Connectivity Rating",
    "Saftey Rating",
]

CATEGORICAL_COLUMNS = [
    "property_type",
    "society",
    "facing",
    "furnishing_status",
    "agePossession",
    "balcony",
]

DROP_AFTER_ENGINEERING = ["bedRoom", "bathroom", "balcony", "agePossession"]

MODEL_FEATURES = [
    "property_type",
    "society",
    "price_per_sqft",
    "area",
    "floorNum",
    "facing",
    "Environment Rating",
    "Lifestyle Rating ",
    "Connectivity Rating",
    "Saftey Rating",
    "furnishing_status",
    "Private Garden / Terrace",
    "Maintenance Staff",
    "Water Storage",
    "Park",
    "Visitor Parking",
    "Waste Disposal",
    "Rain Water Harvesting",
    "Security",
    "total_rooms",
    "house_age",
]

STANDARD_AGE_VALUES = {
    "0 to 1 Year Old",
    "1 to 5 Year Old",
    "5 to 10 Year Old",
    "10+ Year Old",
}

RATING_NAME_MAP = {
    "Environment": "Environment Rating",
    "Lifestyle": "Lifestyle Rating ",
    "Connectivity": "Connectivity Rating",
    "Safety": "Saftey Rating",
}


def parse_list(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if not isinstance(value, str):
        return []

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []

    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def normalize_age(value: Any) -> str:
    if pd.isna(value):
        return "0 to 1 Year Old"

    value = str(value).strip()
    if value in STANDARD_AGE_VALUES:
        return value
    return "0 to 1 Year Old"


def furnishing_status(value: Any) -> str:
    items = parse_list(value)
    if not items:
        return "Unfurnished"

    available_items = [item for item in items if not item.strip().lower().startswith("no ")]
    if len(available_items) >= 5:
        return "Fully Furnished"
    if available_items:
        return "Partially Furnished"
    return "Unfurnished"


def extract_rating(raw_rating: Any, rating_column: str) -> float | None:
    rating_name = next(
        name for name, column in RATING_NAME_MAP.items() if column == rating_column
    )

    for item in parse_list(raw_rating):
        match = re.search(rf"{rating_name}\s*(\d+(?:\.\d+)?)\s*out of 5", item)
        if match:
            return float(match.group(1))
    return None


def sorted_options(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist())


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df["property_type"] = df["property_type"].fillna("Unknown").astype(str)
    df["society"] = df["society"].fillna("Unknown").astype(str)
    df["facing"] = df["facing"].fillna("Unknown").astype(str)
    df["agePossession"] = df["agePossession"].apply(normalize_age)
    df["furnishing_status"] = df["furnishDetails"].apply(furnishing_status)

    for feature in BINARY_FEATURES:
        df[feature] = df["features"].apply(lambda value: int(feature in parse_list(value)))

    for rating in RATING_FEATURES:
        df[rating] = df["rating"].apply(lambda value: extract_rating(value, rating))
        df[rating] = df[rating].fillna(df[rating].median())

    numeric_columns = ["price_per_sqft", "area", "bedRoom", "bathroom", "floorNum"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df[column] = df[column].fillna(df[column].median())

    df["balcony"] = df["balcony"].fillna("0").astype(str)
    return df


def encode_categories(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    encoded_df = df.copy()
    category_mappings = {}

    for column in CATEGORICAL_COLUMNS:
        options = sorted_options(encoded_df[column])
        category_mappings[column] = {value: index for index, value in enumerate(options)}
        encoded_df[column] = encoded_df[column].astype(str).map(category_mappings[column])

    return encoded_df, category_mappings


def prepare_training_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    engineered_df = engineer_features(raw_df)
    engineered_df[TARGET_COLUMN] = pd.to_numeric(
        engineered_df[TARGET_COLUMN], errors="coerce"
    )
    engineered_df = engineered_df.dropna(subset=[TARGET_COLUMN])
    encoded_df, category_mappings = encode_categories(engineered_df)

    encoded_df["total_rooms"] = (
        encoded_df["bedRoom"] + encoded_df["bathroom"] + encoded_df["balcony"]
    )
    encoded_df["house_age"] = encoded_df["agePossession"]

    features = encoded_df.drop(
        columns=[TARGET_COLUMN, *DROP_AFTER_ENGINEERING]
    )[MODEL_FEATURES]
    target = encoded_df[TARGET_COLUMN]
    return features, target, category_mappings


def train_model(features: pd.DataFrame, target: pd.Series) -> tuple[VotingRegressor, dict]:
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
    )

    random_forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    gradient_boosting = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.2,
        max_depth=3,
        random_state=42,
    )
    model = VotingRegressor(
        estimators=[
            ("rf", random_forest),
            ("gb", gradient_boosting),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)

    metrics = {
        "mse": mse,
        "rmse": mse**0.5,
        "mae": mean_absolute_error(y_test, predictions),
        "r2": r2_score(y_test, predictions),
    }
    return model, metrics


def main() -> None:
    raw_df = pd.read_csv(DATA_PATH)
    features, target, category_mappings = prepare_training_data(raw_df)
    model, metrics = train_model(features, target)

    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "model_file": MODEL_PATH.name,
                "training_data_file": DATA_PATH.name,
                "scikit_learn_version": sklearn.__version__,
                "feature_names": MODEL_FEATURES,
                "category_mappings": category_mappings,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metadata: {METADATA_PATH}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
