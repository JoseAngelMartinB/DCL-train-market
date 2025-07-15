import numpy as np
import pandas as pd

def load_data(path_data, y_col="passengers"):
    data = pd.read_csv(path_data, sep=",")
    y = data[y_col]
    X = data.drop(columns=[y_col], axis=1, inplace=False)
    return X, y

def clip_data(y, min_value=0, max_value=5000):
    return np.clip(y, None, max_value)

def sine_cosine_transform(df, col, period, drop=False):
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
    if drop:
        df.drop(columns=[col], inplace=True)
    return df

def process_categorical_features(df, cols_to_encode):
    return pd.get_dummies(df, columns=cols_to_encode, drop_first=False, dtype=int)

def remove_unused_columns(df, columns_to_drop):
    df.drop(columns=columns_to_drop, inplace=True)
    return df

def extract_date_from_service_id(service_id):
    # service_id format: 00003_01-01-2025-06.27
    try:
        date_str = service_id.split('_')[1].split('-')[0:3]
        date_str = '-'.join(date_str)  # '01-01-2025'
        return pd.to_datetime(date_str, format='%d-%m-%Y')
    except Exception:
        return pd.NaT

def is_holiday_pm1(date, holiday_calendar):
    # Check if date, date-1, or date+1 is a holiday
    for offset in [-1, 0, 1]:
        check_date = date + pd.Timedelta(days=offset)
        if check_date in holiday_calendar:
            return 1
    return 0

if __name__ == "__main__":
    path_data = "../DataGenerationROBIN/data/MAD-BCN/aggregated/MAD-BCN_2025.csv"
    X, y = load_data(path_data)

    # --- Add holiday features ---
    import holidays

    X['service_date'] = X['service_id'].apply(extract_date_from_service_id)
    es_holidays = holidays.country_holidays('ES', years=[2025])
    es_holidays_madrid = holidays.country_holidays('ES', years=[2025], subdiv='MD')
    es_holidays_catalonia = holidays.country_holidays('ES', years=[2025], subdiv='CT')

    X['is_holiday_pm1'] = X['service_date'].apply(lambda d: is_holiday_pm1(d, es_holidays))
    X['is_holiday_madrid_pm1'] = X['service_date'].apply(lambda d: is_holiday_pm1(d, es_holidays_madrid))
    X['is_holiday_catalonia_pm1'] = X['service_date'].apply(lambda d: is_holiday_pm1(d, es_holidays_catalonia))

    y = clip_data(y)

    X_processed = X.copy()
    X_processed = sine_cosine_transform(X_processed, "month", 12, drop=True)
    X_processed = sine_cosine_transform(X_processed, "day_of_week", 7, drop=True)
    X_processed = sine_cosine_transform(X_processed, "departure_time", 24, drop=True)

    cols_to_encode = ["train_type"] + [f"train_type_competitor_{i}" for i in range(-2, 3) if i != 0]
    X_processed = process_categorical_features(X_processed, cols_to_encode)

    columns_to_drop = ["service_id", "capacity", "train_model", "year", "service_date"]
    X_processed = remove_unused_columns(X_processed, columns_to_drop)

    # Join X_processed and y into a single DataFrame and save as CSV
    df_clean = X_processed.copy()
    df_clean["passengers"] = y
    df_clean.to_csv("preprocesed_data/cleaned_MAD-BCN_2025.csv", index=False)