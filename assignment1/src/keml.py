import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def data_overview(df):
    """Display a comprehensive overview of the DataFrame"""
    
    print("=" * 60)
    print("📊 DATA OVERVIEW")
    print("=" * 60)
    
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]:,}")
    memory_usage = df.memory_usage(deep=True)
    total_memory_mb = memory_usage.sum() / (1024**2)
    print(f"Total size: {total_memory_mb:.2f} MB\n")
    df.info()
    
    # Call helper functions
    check_missing_data(df)
    check_duplicated_data(df)
    find_duplicate_columns(df)


def check_missing_data(df):
    """Check for missing data in the DataFrame"""
    print("\n🔍 Check missing data")
    print("-" * 30)
    
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent.round(2)
    })
    print(missing_df)


def check_duplicated_data(df):
    """Check for duplicated rows in the DataFrame"""
    print("\n🔍 Check duplicated data")
    print("-" * 30)
    
    # Total duplicates
    total_duplicates = df.duplicated().sum()
    duplicate_percentage = (total_duplicates / len(df)) * 100
    
    print(f"Total duplicate rows: {total_duplicates:,}")
    print(f"Percentage of duplicates: {duplicate_percentage:.2f}%")
    
    if total_duplicates > 0:
        # Show which rows are duplicated
        duplicate_mask = df.duplicated(keep=False)
        duplicate_groups = df[duplicate_mask].groupby(list(df.columns)).size()
        
        print("\nDuplicated rows:")
        print("─" * 30)
        for i, (group, count) in enumerate(duplicate_groups.items()):
            print(f"Group {i+1}: {count} identical rows")
        
    else:
        print("✅ No duplicate rows found!")


def find_duplicate_columns(df):
    """Find and return duplicate column names with their counts and comprehensive data comparison"""
    print("\n🔍 Check if there are duplicated columns")
    print("-" * 40)
    
    column_counts = df.columns.value_counts()
    duplicated_columns = column_counts[column_counts > 1]
    
    if duplicated_columns.empty:
        print("No duplicated column names found.")
        return None
    
    print(f"Duplicated columns count:\n{duplicated_columns}\n")
    
    results = {}
    
    for col_name in duplicated_columns.index:
        print(f"\n📊 Check if duplicated columns have the same data: '{col_name}'")
        print("-" * 50)
        
        # Get all columns with this name
        duplicate_cols = df.loc[:, df.columns == col_name]
        num_duplicates = duplicate_cols.shape[1]
        
        print(f"Number of columns: {num_duplicates}")
        
        # Display sample data
        print("Check 5 random rows:")
        print(duplicate_cols.sample())
        
        # Check for identical data using helper function
        comparison_results = _compare_duplicate_columns(duplicate_cols, num_duplicates)
        
        # Display comparison results
        print("\nData comparison results:")
        for comparison, result in comparison_results.items():
            if "IDENTICAL" in result:
                print(f"  ✅ {comparison}: {result}")
            else:
                print(f"  ❌ {comparison}: {result}")
        
        # Store results
        results[col_name] = {
            'count': num_duplicates,
            'identical': all("IDENTICAL" in result for result in comparison_results.values()),
            'comparison_details': comparison_results
        }
        
        # Show sample differences if data is different
        _show_sample_differences(duplicate_cols, comparison_results, col_name, num_duplicates)
    
    return results


def _compare_duplicate_columns(duplicate_cols, num_duplicates):
    """Helper function to compare duplicate columns for identical data"""
    comparison_results = {}
    
    if num_duplicates > 1:
        first_col = duplicate_cols.iloc[:, 0]
        
        for i in range(1, num_duplicates):
            current_col = duplicate_cols.iloc[:, i]
            
            # Handle NaN values properly
            are_identical = first_col.equals(current_col)
            
            if are_identical:
                comparison_results[f"Col1_vs_Col{i+1}"] = "IDENTICAL"
            else:
                # Count differences (including NaN handling)
                mask_both_notna = first_col.notna() & current_col.notna()
                mask_both_na = first_col.isna() & current_col.isna()
                mask_same_values = (first_col == current_col) & mask_both_notna
                
                identical_count = (mask_same_values | mask_both_na).sum()
                different_count = len(first_col) - identical_count
                
                comparison_results[f"Col1_vs_Col{i+1}"] = f"DIFFERENT ({different_count} differences)"
    
    return comparison_results


def _show_sample_differences(duplicate_cols, comparison_results, col_name, num_duplicates):
    """Helper function to show sample differences between duplicate columns"""
    if comparison_results and not all("IDENTICAL" in result for result in comparison_results.values()):
        print("\nSample of different values:")
        first_col = duplicate_cols.iloc[:, 0]
        for i in range(1, num_duplicates):
            current_col = duplicate_cols.iloc[:, i]
            if not first_col.equals(current_col):
                # Show first few differences
                diff_mask = (first_col != current_col) | (first_col.isna() != current_col.isna())
                if diff_mask.any():
                    sample_size = min(3, diff_mask.sum())
                    diff_indices = diff_mask[diff_mask].index[:sample_size]
                    
                    sample_df = pd.DataFrame({
                        'Index': diff_indices,
                        f'{col_name}_Col1': first_col.loc[diff_indices],
                        f'{col_name}_Col{i+1}': current_col.loc[diff_indices]
                    })
                    print(f"  Differences between Column 1 and Column {i+1}:")
                    print(f"{sample_df.to_string(index=False)}")


def remove_duplicate_columns(df, keep='first', suffix=None):
    """
    Remove or rename duplicate columns
    
    Parameters:
    -----------
    df : DataFrame
        The DataFrame to process
    keep : str, default 'first'
        Which duplicates to keep: 'first', 'last', or 'none'
    suffix : str, optional
        If provided, rename duplicates by adding suffix instead of removing
    inplace : bool, default False
        If True, modify the original DataFrame. If False, return a new DataFrame.
    
    Returns:
    --------
    DataFrame : DataFrame with duplicates handled (if inplace=False)
    dict : Summary of actions taken
    """
    duplicated_mask = df.columns.duplicated(keep=False)
    
    if not duplicated_mask.any():
        print("✅ No duplicate columns to remove")
        return df
    
    if suffix is not None:
        # Rename duplicates with suffix
        return _rename_duplicate_columns(df, suffix)
    else:
        # Remove duplicates
        return _remove_duplicate_columns(df, keep)


def _rename_duplicate_columns(df, suffix):
    """Helper function to rename duplicate columns with suffix"""
    new_columns = []
    column_counts = {}
    
    for col in df.columns:
        if col in column_counts:
            column_counts[col] += 1
            new_columns.append(f"{col}{suffix}{column_counts[col]}")
        else:
            column_counts[col] = 0
            new_columns.append(col)

        result_df = df.copy()
        result_df.columns = new_columns
    
    renamed_count = sum(1 for col in new_columns if suffix in col)
    print(f"✅ Renamed {renamed_count} duplicate columns with suffix '{suffix}'")
    
    return result_df


def _remove_duplicate_columns(df, keep):
    """Helper function to remove duplicate columns"""
    original_shape = df.shape
    print(f"Original shape: {original_shape
                             }")
    if keep == 'none':
        # Remove all duplicates (keep no duplicated columns)
        mask_to_keep = ~df.columns.duplicated(keep=False)
    else:
        # Keep first or last occurrence
        mask_to_keep = ~df.columns.duplicated(keep=keep)
    
    result_df = df.loc[:, mask_to_keep]
    
    removed_count = original_shape[1] - mask_to_keep.sum()
    print(f"✅ Removed {removed_count} duplicate columns (kept '{keep}' occurrence)")
    print(f"Shape after removing duplicates:{result_df.shape}")
    
    return result_df


def sort_columns_alphabetically(df, ascending=True):
    """Sort DataFrame columns alphabetically"""

    sorted_df = df.sort_index(axis=1, ascending=ascending)
    print(f"✅ Columns sorted alphabetically ({'A-Z' if ascending else 'Z-A'})")
    return sorted_df


def analyze_cat_feat(df):
    """
    Analyze categorical features in a dataframe by extracting object-type columns,
    performing value counts, and creating count plots for each feature.
    
    Parameters:
    df (pd.DataFrame): Input dataframe to analyze
    
    Returns:
    dict: Dictionary containing value counts for each categorical feature
    """
    # Extract columns of type 'object' (categorical features)
    cat_columns = df.select_dtypes(include=['object']).columns.tolist()
    
    if not cat_columns:
        print("No categorical (object) columns found in the dataframe.")
        return {}
    
    print(f"Found {len(cat_columns)} categorical features: {cat_columns}")
    
    # Dictionary to store value counts
    value_counts_dict = {}
    
    # Analyze each categorical feature
    for col in cat_columns:
        print(f"\n{'='*50}")
        print(f"Analyzing feature: {col}")
        print(f"{'='*50}")
        
        # Get value counts
        value_counts = df[col].value_counts()
        value_counts_dict[col] = value_counts
        
        # Display value counts
        print(f"\nValue counts for '{col}':")
        print(value_counts)
        print(f"\nUnique values: {df[col].nunique()}")
        print(f"Missing values: {df[col].isnull().sum()}")
        
        # Create count plot
        plt.figure(figsize=(10, 6))
        
        # Handle cases with many unique values
        if df[col].nunique() > 20:
            # Show only top 20 categories
            top_categories = value_counts.head(20)
            plt.figure(figsize=(12, 8))
            sns.countplot(data=df[df[col].isin(top_categories.index)], x=col)
            plt.title(f'Count Plot for {col} (Top 20 categories)')
        else:
            sns.countplot(data=df, x=col)
            plt.title(f'Count Plot for {col}')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
    
    return value_counts_dict