import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# Define paths
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'ecommerce_inventory_demand.csv') 
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'processed_data.pkl')


def load_data(filepath):
    """Load the CSV data file"""
    # Print loading message
    print(f"Loading data from {filepath}...")
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filepath)
    # Convert the 'date' column to datetime format for proper handling
    df['date'] = pd.to_datetime(df['date'])
    # Sort the DataFrame by 'product_id' and 'date' to prepare for time-series operations like forward-fill
    df = df.sort_values(['product_id', 'date']).reset_index(drop=True)
    # Print the number of rows and columns loaded
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    # Return the loaded and sorted DataFrame
    return df


def explore_data(df):
    """Explore the dataset"""
    print("\n=== Data Exploration ===")
    print(f"\nShape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nBasic statistics:\n{df.describe()}")
    return df


def handle_missing_values(df):
    """Handle missing values in the dataset"""
    # Print section header
    print("\n=== Handling Missing Values ===")
    
    # Calculate missing values before handling
    # Get the count of missing values per column
    missing_before = df.isnull().sum()
    # Calculate the percentage of missing values per column
    missing_percent_before = (missing_before / len(df)) * 100
    
    # Print missing values before cleaning
    # Display header for the table
    print("\nMissing values before cleaning:")
    print("-" * 50)
    # Loop through each column to print missing count and percentage
    for col in df.columns:
        if missing_before[col] > 0:
            print(f"{col}: {missing_before[col]} missing ({missing_percent_before[col]:.2f}%)")
        else:
            print(f"{col}: 0 missing (0.00%)")
    
    # Identify columns with missing values
    # List columns that have at least one missing value
    cols_with_missing = missing_before[missing_before > 0].index.tolist()
    # Print the list of columns with missing values
    print(f"\nColumns with missing values: {cols_with_missing}")
    
    # Handle missing values using forward-fill
    # Create a copy of the DataFrame to avoid modifying the original
    df_clean = df.copy()
    # Apply forward-fill to propagate the last valid observation forward
    df_clean = df_clean.ffill()
    # Print message about forward-fill application
    print("\nApplied forward-fill to handle missing values.")
    
    # Correct price outliers using IQR method
    # Define a function to correct outliers for a given column
    def correct_outliers_iqr(df, col):
        # Calculate the first quartile (25th percentile)
        Q1 = df[col].quantile(0.25)
        # Calculate the third quartile (75th percentile)
        Q3 = df[col].quantile(0.75)
        # Calculate the interquartile range
        IQR = Q3 - Q1
        # Determine the lower bound for outliers
        lower_bound = Q1 - 1.5 * IQR
        # Determine the upper bound for outliers
        upper_bound = Q3 + 1.5 * IQR
        # Clip the values in the column to the bounds, correcting outliers
        df[col] = df[col].clip(lower_bound, upper_bound)
        # Return the modified DataFrame
        return df
    
    # Apply outlier correction to the 'unit_price' column
    df_clean = correct_outliers_iqr(df_clean, 'unit_price')
    # Print message about outlier correction
    print("Corrected outliers in 'unit_price' using IQR method.")
    
    # Calculate missing values after handling
    # Get the count of missing values per column after cleaning
    missing_after = df_clean.isnull().sum()
    # Calculate the percentage of missing values per column after cleaning
    missing_percent_after = (missing_after / len(df_clean)) * 100
    
    # Print missing values after cleaning
    # Display header for the table
    print("\n\nMissing values after cleaning:")
    print("-" * 50)
    # Loop through each column to print missing count and percentage
    for col in df.columns:
        if missing_after[col] > 0:
            print(f"{col}: {missing_after[col]} missing ({missing_percent_after[col]:.2f}%)")
        else:
            print(f"{col}: 0 missing (0.00%)")
    
    # Check for remaining missing values
    # List columns that still have missing values
    remaining_missing_cols = missing_after[missing_after > 0].index.tolist()
    if remaining_missing_cols:
        # Print warning if there are still missing values
        print(f"\nWarning: Still have missing values in columns: {remaining_missing_cols}")
    else:
        # Print success message if all missing values are handled
        print("\n✓ All missing values have been handled successfully!")
    
    # Print the change in dataset shape
    print(f"\nDataset shape: {df.shape} -> {df_clean.shape}")
    
    # Return the cleaned DataFrame
    return df_clean


def feature_engineering(df):
    """Create new features from existing data"""
    # Print section header for feature engineering
    print("\n=== Feature Engineering ===")
    
    # Extract date features
    # Extract year from the date column
    df['year'] = df['date'].dt.year
    # Extract quarter from the date column
    df['quarter'] = df['date'].dt.quarter
    # Extract month from the date column
    df['month'] = df['date'].dt.month
    # Extract day of month from the date column
    df['day_of_month'] = df['date'].dt.day
    # Extract week of year from the date column
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    # Extract day of week as name from the date column
    df['day_of_week'] = df['date'].dt.day_name()
    
    # Create lag features for units_sold
    # Create lag-7 feature: shift units_sold by 7 days within each product group
    df['units_sold_lag_7'] = df.groupby('product_id')['units_sold'].shift(7)
    # Create lag-14 feature: shift units_sold by 14 days within each product group
    df['units_sold_lag_14'] = df.groupby('product_id')['units_sold'].shift(14)
    
    # Create rolling statistics for units_sold
    # Calculate 7-day rolling mean of units_sold within each product group
    df['units_sold_rolling_mean_7'] = df.groupby('product_id')['units_sold'].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    # Calculate 7-day rolling standard deviation of units_sold within each product group
    df['units_sold_rolling_std_7'] = df.groupby('product_id')['units_sold'].transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).std())
    # Calculate 30-day rolling mean of units_sold within each product group
    df['units_sold_rolling_mean_30'] = df.groupby('product_id')['units_sold'].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean())
    # Calculate 30-day rolling standard deviation of units_sold within each product group
    df['units_sold_rolling_std_30'] = df.groupby('product_id')['units_sold'].transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).std())
    
    # Create weekend indicator
    # Check if day_of_week is Saturday or Sunday and convert to integer
    df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    
    # Note: is_promotion column is already present in the dataset


### Why i need this code? ###
#In demand forecasting, patterns often repeat cyclically (e.g., higher sales at month-end, seasonal spikes).
#Without this, model might miss these patterns or learn wrong relationships.
#It's a standard technique in time-series ML—think of it as "tricking" the model into seeing time as a loop, not a line.
   
   
    # Create cyclical encodings for temporal features
    # Cyclical encoding for day_of_month (1-31)
    # Calculate sine component for day of month
    df['day_of_month_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
    # Calculate cosine component for day of month
    df['day_of_month_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
    # Cyclical encoding for month (1-12)
    # Calculate sine component for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    # Calculate cosine component for month
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    # Cyclical encoding for quarter (1-4)
    # Calculate sine component for quarter
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    # Calculate cosine component for quarter
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4) #
    
    # Create price-related features
    # Categorize unit_price into 5 bins with labels
    df['price_category'] = pd.cut(df['unit_price'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Stock level features
    # Calculate stock ratio as stock_on_hand divided by reorder_point plus one
    df['stock_ratio'] = df['stock_on_hand'] / (df['reorder_point'] + 1)
    # Determine if reorder is needed (1 if stock <= reorder_point, else 0)
    df['needs_reorder'] = (df['stock_on_hand'] <= df['reorder_point']).astype(int)
    
    # Print the new shape of the DataFrame after adding features
    print(f"Added new features. New shape: {df.shape}")
    # Return the modified DataFrame
    return df


def encode_categorical(df):
    """Encode categorical variables"""
    # Print section header
    print("\n=== Encoding Categorical Variables ===")
    
    # One-hot encode product_category
    # Create dummy variables for product_category
    product_category_dummies = pd.get_dummies(df['product_category'], prefix='category')
    # Concatenate the dummies to the DataFrame
    df = pd.concat([df, product_category_dummies], axis=1)
    # Print the categories
    print(f"Product categories one-hot encoded: {list(product_category_dummies.columns)}")
    
    # One-hot encode day_of_week
    # Create dummy variables for day_of_week
    day_of_week_dummies = pd.get_dummies(df['day_of_week'], prefix='day')
    # Concatenate the dummies to the DataFrame
    df = pd.concat([df, day_of_week_dummies], axis=1)
    # Print the days
    print(f"Day of week one-hot encoded: {list(day_of_week_dummies.columns)}")
    
    # Label encode product_id
    # Use LabelEncoder for product_id
    le_product = LabelEncoder()
    df['product_id_encoded'] = le_product.fit_transform(df['product_id'])
    # Print the number of unique products
    print(f"Number of unique products: {len(le_product.classes_)}")
    
    # Encode price_category
    if 'price_category' in df.columns:
        # Use LabelEncoder for price_category
        le_price = LabelEncoder()
        df['price_category_encoded'] = le_price.fit_transform(df['price_category'].astype(str))
    
    # Return the modified DataFrame
    return df


def select_features(df, target_col='units_sold'):
    """Select features for model training"""
    # Print section header
    print("\n=== Feature Selection ===")
    
    # Find non-numeric columns to drop
    # Select columns that are not numeric (e.g., object, datetime, category)
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    print("=" * 50)
    # Print the non-numeric columns
    print(f"Non-numeric columns to drop: {non_numeric_cols}")
    print("=" * 50)
    print(f"Target column: {target_col}")
    print("=" * 50)
    # Add the target column to the drop list
    columns_to_drop = non_numeric_cols + [target_col]
    
    # Keep only numeric columns (including one-hot encoded dummies which are numeric)
    # Select columns that are not in the drop list
    feature_cols = [col for col in df.columns if col not in columns_to_drop]
    
    # Print selected features
    print(f"Selected features: {feature_cols}")
    
    # Return the list of feature columns
    return feature_cols


def split_data(df, feature_cols, target_col='units_sold', test_size=0.2):
    """Split data into train and test sets"""
    from sklearn.model_selection import train_test_split
    
    X = df[feature_cols]
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """Scale numerical features"""
    # Print section header
    print("\n=== Scaling Features ===")
    
    # Import MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    # Initialize MinMaxScaler
    scaler = MinMaxScaler()  
    """
        MinMaxScaler Formula:
        The MinMaxScaler transforms features by scaling each feature to a given range, typically [0, 1].
        The formula is:

        scaled_value = (X - X_min) / (X_max - X_min)

        Where:
        - X is the original value of the feature
        - X_min is the minimum value of the feature in the training set
        - X_max is the maximum value of the feature in the training set
        - The resulting scaled_value will be between 0 and 1
    """
    # Fit and transform training features
    X_train_scaled = scaler.fit_transform(X_train) 
    # Transform test features
    X_test_scaled = scaler.transform(X_test)
    
    # Print success message
    print("Features scaled successfully with MinMaxScaler")
    
    # Return scaled features and scaler
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline():
    """Main preprocessing pipeline"""
    print("=" * 50)
    print("Starting Data Preprocessing Pipeline")
    print("=" * 50)
    
    # Load data
    df = load_data(DATA_PATH)
    
    # Explore data
    df = explore_data(df)
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Encode categorical
    df = encode_categorical(df)
    
    # Select features
    feature_cols = select_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(df, feature_cols)
    
    # Scale demand (target) with MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    demand_scaler = MinMaxScaler()
    y_train_scaled = demand_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten() # 
    y_test_scaled = demand_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
    print("Demand (units_sold) scaled with MinMaxScaler")

    """
    What reshape(-1, 1) does:
    -1 means "infer the number of rows" (so it keeps all 1000 samples).
    1 means 1 column (since it's a single target variable).
    Result: Transforms (1000,) → (1000, 1).
    Why flatten() after?: fit_transform returns a 2D array, but want the target back as 1D for model training, so flatten() converts (1000, 1) back to (1000,).
    """
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("\n" + "=" * 50)
    print("Preprocessing Complete!")
    print("=" * 50)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_scaled,
        'y_test': y_test_scaled,
        'y_train_original': y_train,
        'y_test_original': y_test,
        'feature_cols': feature_cols,
        'scaler': scaler,
        'demand_scaler': demand_scaler,
        'df': df
    }


if __name__ == "__main__":
    # Run preprocessing
    results = preprocess_pipeline()
    
    # Save processed data
    print("\nSaving processed data...")
    import pickle
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(results, f)
    print(f"Data saved to {OUTPUT_PATH}")
    
    # Save cleaned data to cleaned_data folder
    cleaned_csv_path = os.path.join(os.path.dirname(__file__), '..', '..', 'cleaned_data', 'cleaned_data.csv')
    results['df'].to_csv(cleaned_csv_path, index=False)
    print(f"Cleaned data saved to {cleaned_csv_path}")