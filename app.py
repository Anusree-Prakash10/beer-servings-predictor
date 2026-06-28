import joblib
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Beer Servings Alcohol Predictor", page_icon="🍺", layout="centered")

st.title("Beer Servings Alcohol Predictor")
st.write("Estimate the total litres of pure alcohol from beer, spirit, wine servings, and continent.")
st.image(
    "https://images.unsplash.com/photo-1514362545857-3bc16c4c7d1b?auto=format&fit=crop&w=900&q=80",
    use_container_width=True,
)

MODEL_PATH = Path(__file__).parent / "beer_alcohol_model.joblib"
DATA_PATH = Path(__file__).parent / "beer-servings (1).csv"


@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    df = pd.read_csv(DATA_PATH)
    features = ['beer_servings', 'spirit_servings', 'wine_servings', 'continent']
    target = 'total_litres_of_pure_alcohol'

    df = df.dropna(subset=[target]).copy()
    X = df[features]
    y = df[target]

    numeric_features = ['beer_servings', 'spirit_servings', 'wine_servings']
    categorical_features = ['continent']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=300, random_state=42)),
        ]
    )
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model


model = load_model()

st.subheader("Enter the beer-serving details")
beer_servings = st.number_input("Beer servings", min_value=0.0, max_value=500.0, value=120.0, step=1.0)
spirit_servings = st.number_input("Spirit servings", min_value=0.0, max_value=500.0, value=80.0, step=1.0)
wine_servings = st.number_input("Wine servings", min_value=0.0, max_value=500.0, value=50.0, step=1.0)
continent = st.selectbox("Continent", ["Africa", "Asia", "Europe", "North America", "South America", "Oceania"])

if st.button("Predict"):
    input_df = pd.DataFrame(
        [{
            'beer_servings': beer_servings,
            'spirit_servings': spirit_servings,
            'wine_servings': wine_servings,
            'continent': continent,
        }]
    )
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted total litres of pure alcohol: {prediction:.2f}")
    st.caption("The model is trained on the beer-servings dataset using regression.")
