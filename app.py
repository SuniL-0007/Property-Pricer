import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from train_model import (
    BINARY_FEATURES,
    DROP_AFTER_ENGINEERING,
    MODEL_FEATURES,
    RATING_FEATURES,
    TARGET_COLUMN,
    engineer_features,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "house_cleaned.csv"
MODEL_PATH = BASE_DIR / "ensemble_model_best.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return engineer_features(pd.read_csv(DATA_PATH))


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def sorted_options(series: pd.Series) -> list[str]:
    return sorted(series.dropna().astype(str).unique().tolist())


def category_options(
    df: pd.DataFrame, column: str, metadata: dict
) -> tuple[list[str], dict[str, int]]:
    mapping = metadata.get("category_mappings", {}).get(column)
    if mapping:
        return list(mapping.keys()), {key: int(value) for key, value in mapping.items()}

    options = sorted_options(df[column])
    return options, {value: index for index, value in enumerate(options)}


def default_numeric(df: pd.DataFrame, column: str) -> float:
    return float(df[column].median())


def numeric_bounds(df: pd.DataFrame, column: str) -> tuple[float, float]:
    return float(df[column].min()), float(df[column].max())


def build_user_input(df: pd.DataFrame, metadata: dict) -> dict:
    property_type_options, property_type_mapping = category_options(
        df, "property_type", metadata
    )
    society_options, society_mapping = category_options(df, "society", metadata)
    facing_options, facing_mapping = category_options(df, "facing", metadata)
    balcony_options, balcony_mapping = category_options(df, "balcony", metadata)
    age_options, age_mapping = category_options(df, "agePossession", metadata)
    furnishing_options, furnishing_mapping = category_options(
        df, "furnishing_status", metadata
    )

    defaults = {
        "property_type": str(df["property_type"].mode().iat[0]),
        "society": str(df["society"].mode().iat[0]),
        "facing": str(df["facing"].mode().iat[0]),
        "balcony": str(df["balcony"].mode().iat[0]),
        "agePossession": str(df["agePossession"].mode().iat[0]),
        "furnishing_status": str(df["furnishing_status"].mode().iat[0]),
    }

    user_input: dict[str, float | int | str] = {}

    st.subheader("Property Basics")
    left, right = st.columns(2)
    with left:
        user_input["property_type"] = st.selectbox(
            "Property category",
            property_type_options,
            index=property_type_options.index(defaults["property_type"]),
        )
        user_input["society"] = st.selectbox(
            "Society",
            society_options,
            index=society_options.index(defaults["society"]),
        )
        min_area, max_area = numeric_bounds(df, "area")
        user_input["area"] = st.number_input(
            "Area (sq ft)",
            min_value=max(1.0, min_area),
            max_value=max_area,
            value=default_numeric(df, "area"),
            step=50.0,
        )
        user_input["bedRoom"] = st.number_input(
            "Bedrooms",
            min_value=int(df["bedRoom"].min()),
            max_value=int(df["bedRoom"].max()),
            value=int(df["bedRoom"].median()),
            step=1,
        )
        user_input["bathroom"] = st.number_input(
            "Bathrooms",
            min_value=int(df["bathroom"].min()),
            max_value=int(df["bathroom"].max()),
            value=int(df["bathroom"].median()),
            step=1,
        )
        user_input["balcony"] = st.selectbox(
            "Balconies",
            balcony_options,
            index=balcony_options.index(defaults["balcony"]),
        )

    with right:
        min_pps, max_pps = numeric_bounds(df, "price_per_sqft")
        user_input["price_per_sqft"] = st.number_input(
            "Price per sq ft",
            min_value=max(1, int(min_pps)),
            max_value=int(max_pps),
            value=int(df["price_per_sqft"].median()),
            step=500,
        )
        user_input["floorNum"] = st.number_input(
            "Floor number",
            min_value=int(df["floorNum"].min()),
            max_value=int(df["floorNum"].max()),
            value=int(df["floorNum"].median()),
            step=1,
        )
        user_input["facing"] = st.selectbox(
            "Facing direction",
            facing_options,
            index=facing_options.index(defaults["facing"]),
        )
        user_input["agePossession"] = st.selectbox(
            "Property age",
            age_options,
            index=age_options.index(defaults["agePossession"]),
        )
        user_input["furnishing_status"] = st.selectbox(
            "Furnishing",
            furnishing_options,
            index=furnishing_options.index(defaults["furnishing_status"]),
        )

    st.subheader("Ratings")
    rating_cols = st.columns(4)
    for index, column in enumerate(RATING_FEATURES):
        label = column.strip()
        with rating_cols[index]:
            user_input[column] = st.slider(
                label,
                min_value=int(df[column].min()),
                max_value=int(df[column].max()),
                value=int(df[column].median()),
            )

    st.subheader("Amenities")
    amenity_cols = st.columns(4)
    for index, feature in enumerate(BINARY_FEATURES):
        default_value = bool(round(default_numeric(df, feature)))
        with amenity_cols[index % 4]:
            user_input[feature] = int(st.checkbox(feature, value=default_value))

    user_input["property_type"] = property_type_mapping[
        str(user_input["property_type"])
    ]
    user_input["society"] = society_mapping[str(user_input["society"])]
    user_input["facing"] = facing_mapping[str(user_input["facing"])]
    user_input["balcony"] = balcony_mapping[str(user_input["balcony"])]
    user_input["agePossession"] = age_mapping[str(user_input["agePossession"])]
    user_input["furnishing_status"] = furnishing_mapping[
        str(user_input["furnishing_status"])
    ]

    return user_input


def prepare_features(user_input: dict) -> pd.DataFrame:
    input_df = pd.DataFrame([user_input])
    input_df["total_rooms"] = (
        input_df["bedRoom"].astype(float)
        + input_df["bathroom"].astype(float)
        + input_df["balcony"].astype(float)
    )
    input_df["house_age"] = input_df["agePossession"].astype(float)
    input_df = input_df.drop(columns=DROP_AFTER_ENGINEERING)
    return input_df[MODEL_FEATURES]


def show_market_context(df: pd.DataFrame, metadata: dict) -> None:
    st.sidebar.header("Dataset Snapshot")
    st.sidebar.metric("Listings", f"{len(df):,}")
    st.sidebar.metric("Median price", f"Rs {df[TARGET_COLUMN].median():.2f} Cr")
    st.sidebar.metric("Median area", f"{df['area'].median():,.0f} sq ft")
    if metadata.get("scikit_learn_version"):
        st.sidebar.caption(
            f"Model exported with scikit-learn {metadata['scikit_learn_version']}"
        )

    with st.sidebar.expander("Price distribution"):
        st.dataframe(
            df[TARGET_COLUMN].describe(percentiles=[0.25, 0.5, 0.75]).to_frame(
                "price_cr"
            ),
            width="stretch",
        )


def show_comparable_listings(df: pd.DataFrame, user_input: dict) -> None:
    comparable_df = df.copy()
    comparable_df["area_gap"] = (comparable_df["area"] - float(user_input["area"])).abs()
    comparable_df["price_per_sqft_gap"] = (
        comparable_df["price_per_sqft"] - float(user_input["price_per_sqft"])
    ).abs()
    comparable_df = comparable_df.sort_values(["area_gap", "price_per_sqft_gap"]).head(5)

    st.subheader("Closest Listings In The Dataset")
    st.dataframe(
        comparable_df[
            [
                "society",
                "price",
                "price_per_sqft",
                "area",
                "bedRoom",
                "bathroom",
                "balcony",
                "facing",
                "agePossession",
                "furnishing_status",
            ]
        ],
        hide_index=True,
        width="stretch",
    )


def main() -> None:
    st.set_page_config(
        page_title="Property Price Predictor",
        layout="wide",
    )

    st.title("Gurgaon Property Price Predictor")
    st.caption("Estimate property prices in crores using the trained ensemble model.")

    df = load_dataset()
    metadata = load_model_metadata()
    show_market_context(df, metadata)

    try:
        model = load_model()
        model_error = None
    except Exception as exc:  # noqa: BLE001
        model = None
        model_error = exc

    if model_error:
        st.error(
            "The model file could not be loaded. Run `python train_model.py` in this "
            "environment to regenerate a compatible model, then restart Streamlit."
        )
        st.code(str(model_error), language="text")

    with st.form("prediction_form"):
        user_input = build_user_input(df, metadata)
        submitted = st.form_submit_button(
            "Predict Price",
            type="primary",
            disabled=model is None,
        )

    feature_df = prepare_features(user_input)

    if submitted and model is not None:
        prediction = float(model.predict(feature_df)[0])
        lower_estimate = max(0, prediction * 0.9)
        upper_estimate = prediction * 1.1

        result_cols = st.columns(3)
        result_cols[0].metric("Predicted price", f"Rs {prediction:.2f} Cr")
        result_cols[1].metric("Conservative range", f"Rs {lower_estimate:.2f} Cr")
        result_cols[2].metric("Upper range", f"Rs {upper_estimate:.2f} Cr")

        with st.expander("Model input sent for prediction"):
            st.dataframe(feature_df, hide_index=True, width="stretch")

    show_comparable_listings(df, user_input)


if __name__ == "__main__":
    main()
