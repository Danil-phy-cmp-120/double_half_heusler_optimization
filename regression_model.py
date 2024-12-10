import pandas as pd
import numpy as np
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

if __name__ == '__main__':
    # Load data
    data_file = "carrier_transport_data.csv"
    data_df = pd.read_csv(data_file)

    # Reduce dataset size for quick testing
    data_df = data_df.sample(n=100, random_state=42)  # Adjust 'n' as needed

    # Target columns
    target_columns = [
        "Bandgap",
        "Seebeck Coefficient n (S.n.value)",
        "Thermal Conductivity n (kappa.n.value)",
        "Electrical Conductivity n (sigma.n.value)"
    ]

    # Featurization
    data_df = StrToComposition().featurize_dataframe(data_df, "Formula", ignore_errors=True)
    ep_feat = ElementProperty.from_preset(preset_name="magpie", impute_nan=True)
    data_df = ep_feat.featurize_dataframe(data_df, col_id="composition")

    # Retain only columns containing 'mean' and 'volume'
    columns_to_retain = [col for col in data_df.columns if "mean" in col] + ["Volume (V)"]
    data_df = data_df[columns_to_retain + target_columns]

    # Save the featurized dataset to a CSV file
    featurized_csv_file = "filtered_featurized_carrier_transport_data.csv"
    data_df.to_csv(featurized_csv_file, index=False)
    print(f"Filtered featurized data saved to {featurized_csv_file}")

    # Drop rows with missing target values
    data_df = data_df.dropna(subset=target_columns)

    # Separate features and targets
    X = data_df.drop(columns=target_columns)
    y = data_df[target_columns]

    # Split data into training and testing sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=200, max_depth=40, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)


    # Calculate train errors
    train_errors = 100 * (model.predict(X_train) - y_train) / y_train


    # Get feature importances and sort them
    importances = model.feature_importances_
    included = X_train.columns.values
    indices = np.argsort(importances)[::-1]
    sorted_feature_importances = [(included[i], importances[i]) for i in indices]

    # Calculate RMSE and R² for train set
    rmse_train = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
    r2_train = r2_score(y_train, model.predict(X_train))


    # Save results
    results = {
        'train_errors': train_errors.tolist(),
        'sorted_feature_importances': sorted_feature_importances,
        'rmse_train': rmse_train,
        'r2_train': r2_train,
        'test_rmse': test_rmse,
        'test_r2': test_r2
    }

    with open("results_carrier_transport.pickle", "wb") as f:
        pickle.dump(results, f)

    # Save the model
    with open("model_carrier_transport.pickle", "wb") as f:
        pickle.dump(model, f)

    print("Training completed. Model and results saved.")