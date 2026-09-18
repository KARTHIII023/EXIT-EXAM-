from pathlib import Path
import pickle

import pandas as pd
from flask import Flask, flash, redirect, render_template, request, url_for


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATHS = (BASE_DIR / "model.pkl", BASE_DIR / "models.pkl")
FEATURE_COLUMNS_PATH = BASE_DIR / "feature_columns.pkl"


def load_artifacts():
	model_path = next((path for path in MODEL_PATHS if path.exists()), None)
	if model_path is None:
		raise FileNotFoundError("No model artifact found. Expected model.pkl or models.pkl.")
	if not FEATURE_COLUMNS_PATH.exists():
		raise FileNotFoundError("feature_columns.pkl is missing.")

	with model_path.open("rb") as model_file:
		model = pickle.load(model_file)
	with FEATURE_COLUMNS_PATH.open("rb") as columns_file:
		feature_columns = pickle.load(columns_file)

	feature_columns = list(feature_columns)
	model_columns = list(getattr(model, "feature_names_in_", feature_columns))
	if model_columns != feature_columns:
		raise ValueError("feature_columns.pkl does not match the model's feature_names_in_.")
	if getattr(model, "n_features_in_", len(feature_columns)) != len(feature_columns):
		raise ValueError("The model feature count does not match feature_columns.pkl.")

	categorical_columns = [column for column in feature_columns if column.startswith("type_2_")]
	numeric_columns = [column for column in feature_columns if column not in categorical_columns]
	if not categorical_columns:
		raise ValueError("The serialized feature metadata does not contain the expected categorical block.")

	return model, feature_columns, numeric_columns, categorical_columns


model, feature_columns, numeric_columns, categorical_columns = load_artifacts()
category_options = [column.removeprefix("type_2_") for column in categorical_columns]

app = Flask(__name__)
app.config["SECRET_KEY"] = "prediction-app-local-key"


def display_name(name):
	return name.replace("_", " ").title()


def build_input_frame(form):
	values = {}
	for column in numeric_columns:
		raw_value = form.get(column, "").strip()
		if not raw_value:
			raise ValueError(f"{display_name(column)} is required.")
		try:
			values[column] = float(raw_value)
		except ValueError as error:
			raise ValueError(f"{display_name(column)} must be a number.") from error

	selected_category = form.get("type_2", "")
	if selected_category not in {"", *category_options}:
		raise ValueError("Choose a valid secondary type.")
	for column in categorical_columns:
		values[column] = float(column == f"type_2_{selected_category}")

	# DataFrame construction with the serialized columns prevents accidental reordering.
	return pd.DataFrame([values], columns=feature_columns, dtype="float64")


@app.get("/")
def index():
	return render_template(
		"index.html",
		numeric_columns=numeric_columns,
		category_options=category_options,
		display_name=display_name,
	)


@app.post("/predict")
def predict():
	try:
		input_frame = build_input_frame(request.form)
		prediction = model.predict(input_frame)[0]
		probability = None
		if hasattr(model, "predict_proba"):
			probability = float(model.predict_proba(input_frame).max(axis=1)[0])
		return render_template(
			"result.html",
			prediction=str(prediction),
			probability=probability,
			input_values=request.form,
		)
	except (ValueError, TypeError, KeyError) as error:
		flash(str(error), "error")
		return redirect(url_for("index"))
	except Exception:
		app.logger.exception("Prediction failed")
		flash("The prediction could not be completed. Check the input and try again.", "error")
		return redirect(url_for("index"))


if __name__ == "__main__":
	app.run(debug=True)
