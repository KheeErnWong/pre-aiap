import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

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
    print(f"Original shape: {original_shape}")
    if keep == 'none':
        # Remove all duplicates (keep no duplicated columns)
        mask_to_keep = ~df.columns.duplicated(keep=False)
    else:
        # Keep first or last occurrence
        mask_to_keep = ~df.columns.duplicated(keep=keep)
    
    result_df = df.loc[:, mask_to_keep]
    
    removed_count = original_shape[1] - mask_to_keep.sum()
    print(f"✅ Removed {removed_count} duplicate columns (kept '{keep}' occurrence)")
    print(f"Shape after removing duplicates: {result_df.shape}")
    
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
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not cat_columns:
        print("No categorical (object) columns found in the dataframe.")
        return {}
    
    print(f"Found {len(cat_columns)} categorical features (sorted by unique values): {cat_columns}")
    
    # Dictionary to store value counts
    value_counts_dict = {}
    
    # PHASE 1: Display all value counts and statistics (in sorted order)
    print("\n" + "="*80)
    print("CATEGORICAL FEATURES VALUE COUNTS")
    print("="*80)
    
    for col in cat_columns:
        print(f"\n{'='*50}")
        print(f"Feature: {col}")
        print(f"{'='*50}")
        
        # Get value counts and percentages
        value_counts = df[col].value_counts()
        value_percentages = df[col].value_counts(normalize=True) * 100
        
        # Create combined DataFrame with counts and percentages
        value_analysis = pd.DataFrame({
            'Count': value_counts,
            'Percentage': value_percentages.round(2)
        })
        
        value_counts_dict[col] = value_analysis
        
        # Display statistics
        print(f"Unique values: {df[col].nunique()}")
        print(f"Missing values: {df[col].isnull().sum()}")
        
        # Display value counts with percentages
        if df[col].nunique() > 15:
            print("(Showing top 15 categories)")
            print(value_analysis.head(15))
        else:
            print(value_analysis)
    
    # PHASE 2: Create visualizations in rows of 3 (in sorted order)
    print(f"\n{'='*80}")
    print("Categorical Features Distribution")
    print("="*80)
    
    # Set up the color palette
    sns.set_palette("husl")
    
    # Calculate number of rows needed
    n_features = len(cat_columns)
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols  # Ceiling division
    
    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 9*n_rows))
    
    # Flatten axes array for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes = axes.flatten()
    
    # Create plots in sorted order
    for idx, col in enumerate(cat_columns):
        ax = axes[idx]

        # Get value counts for ordering
        value_counts = df[col].value_counts()
        
        # Handle cases with many unique values
        if df[col].nunique() > 15:
            # Show only top 15 categories
            top_categories = df[col].value_counts().head(15)
            plot_data = df[df[col].isin(top_categories.index)]
            order = top_categories.index.tolist()
            title = f'{col}\n(Top 15 categories)\nUnique values: {df[col].nunique()}'
        else:
            plot_data = df
            order = value_counts.index.tolist()
            title = f'{col}\nUnique values: {df[col].nunique()}'
        
        # Create count plot 
        sns.countplot(data=plot_data, x=col, hue=col, ax=ax, palette="husl", legend=False, order=order)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel('')
        
        # Rotate x-axis labels to 90 degrees
        ax.tick_params(axis='x', rotation=90)
        
        # Add value labels on bars (optional)
        for container in ax.containers:
            ax.bar_label(container, fontsize=8, rotation=90, padding=3)
    
    # Hide empty subplots
    for idx in range(len(cat_columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return value_counts_dict


def analyze_num_feat(df):
    """
    Analyze numerical features (int64 columns) in a DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame
    
    Returns:
    dict: Dictionary containing analysis results
    """
    # Extract int64 columns
    numeric_cols = df.select_dtypes(include=['int64']).columns.tolist()
    
    if not numeric_cols:
        print("No int64 columns found in the DataFrame.")
        return None
    
    print(f"Found {len(numeric_cols)} numerical columns: {numeric_cols}\n")
    
    # Store results
    results = {}
    
    # 1. Descriptive Statistics
    print("="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    desc_stats = df[numeric_cols].describe()
    print(desc_stats)
    results['descriptive_stats'] = desc_stats
    
    # Additional statistics
    print("\n" + "="*60)
    print("ADDITIONAL STATISTICS")
    print("="*60)
    additional_stats = pd.DataFrame()
    for col in numeric_cols:
        additional_stats[col] = [
            df[col].skew(),
            df[col].kurtosis(),
            df[col].var(),
        ]
    additional_stats.index = ['Skewness', 'Kurtosis', 'Variance']
    print(additional_stats)
    results['additional_stats'] = additional_stats
    
    # 2. Distribution Analysis with Histograms
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3  # 3 plots per row
    
    plt.figure(figsize=(15, 5 * n_rows))
    plt.suptitle('Distribution of Numerical Features', fontsize=16, y=0.98)
    
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, 3, i)
        
        # Histogram with KDE
        plt.hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        
        # Add KDE curve
        try:
            sns.kdeplot(data=df[col], color='red', linewidth=2)
        except:
            pass
        
        plt.title(f'{col}\n(Mean: {df[col].mean():.2f}, Std: {df[col].std():.2f})')
        plt.xlabel(col)
        plt.ylabel('Density')
        plt.grid(True, alpha=0.3)
        
        # Add statistics text
        plt.text(0.02, 0.98, f'Skew: {df[col].skew():.2f}\nKurt: {df[col].kurtosis():.2f}', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 3. Outlier Detection with Box Plots
    plt.figure(figsize=(15, 5 * n_rows))
    plt.suptitle('Outlier Detection - Box Plots', fontsize=16, y=0.98)
    
    outlier_summary = {}
    
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(n_rows, 3, i)
        
        # Box plot
        box_plot = plt.boxplot(df[col], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        
        plt.title(f'{col}')
        plt.ylabel('Values')
        plt.grid(True, alpha=0.3)
        
        # Calculate outliers using IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(df)) * 100
        
        outlier_summary[col] = {
            'count': outlier_count,
            'percentage': outlier_percentage,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        # Add outlier information
        plt.text(0.02, 0.98, f'Outliers: {outlier_count} ({outlier_percentage:.1f}%)', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    # 4. Outlier Summary
    print("\n" + "="*60)
    print("OUTLIER SUMMARY (IQR Method)")
    print("="*60)
    outlier_df = pd.DataFrame(outlier_summary).T
    print(outlier_df.round(2))
    results['outlier_summary'] = outlier_df
    
    # 5. Correlation Matrix
    if len(numeric_cols) > 1:
        print("\n" + "="*60)
        print("CORRELATION MATRIX")
        print("="*60)
        corr_matrix = df[numeric_cols].corr()
        print(corr_matrix.round(3))
        
        # Correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   mask=mask, square=True, fmt='.2f')
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.show()
        
        results['correlation_matrix'] = corr_matrix
    
    # 6. Zero and Constant Value Analysis
    print("\n" + "="*60)
    print("ZERO AND CONSTANT VALUE ANALYSIS")
    print("="*60)
    zero_constant_analysis = {}
    for col in numeric_cols:
        zero_count = (df[col] == 0).sum()
        zero_percentage = (zero_count / len(df)) * 100
        unique_values = df[col].nunique()
        is_constant = unique_values == 1
        
        zero_constant_analysis[col] = {
            'zero_count': zero_count,
            'zero_percentage': zero_percentage,
            'unique_values': unique_values,
            'is_constant': is_constant
        }
    
    zero_constant_df = pd.DataFrame(zero_constant_analysis).T
    print(zero_constant_df.round(2))
    results['zero_constant_analysis'] = zero_constant_df
    
    return results
