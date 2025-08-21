class DataOverview:
    def __init__(self, df):
        self.df = df
    
    def find_duplicate_columns(self):
        """Find and return duplicate column names with their counts"""
        column_counts = self.df.columns.value_counts()
        print(f"Current df shape: {self.shape}")
        print(f"Duplicated columns count: {column_counts[column_counts > 1]}")

    def remove_duplicate_columns(self, keep='first', suffix=None, inplace=False):
        """
        Remove or rename duplicate columns
        
        Parameters:
        -----------
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
        original_shape = self.df.shape
        duplicated_mask = self.df.columns.duplicated(keep=False)
        
        if not duplicated_mask.any():
            print("✅ No duplicate columns to remove")
            return self.df if not inplace else None, {"action": "none", "removed_columns": 0}
        
        if suffix is not None:
            # Rename duplicates with suffix
            return self._rename_duplicate_columns(suffix, inplace)
        else:
            # Remove duplicates
            return self._remove_duplicate_columns(keep, inplace, original_shape)

    def _rename_duplicate_columns(self, suffix, inplace):
        """Helper method to rename duplicate columns with suffix"""
        new_columns = []
        column_counts = {}
        
        for col in self.df.columns:
            if col in column_counts:
                column_counts[col] += 1
                new_columns.append(f"{col}{suffix}{column_counts[col]}")
            else:
                column_counts[col] = 0
                new_columns.append(col)
        
        if inplace:
            self.df.columns = new_columns
            result_df = None
        else:
            result_df = self.df.copy()
            result_df.columns = new_columns
        
        renamed_count = sum(1 for col in new_columns if suffix in col)
        print(f"✅ Renamed {renamed_count} duplicate columns with suffix '{suffix}'")
        
        return result_df, {
            "action": "renamed",
            "renamed_columns": renamed_count,
            "suffix": suffix
        }
    
    def _remove_duplicate_columns(self, keep, inplace, original_shape):
        """Helper method to remove duplicate columns"""
        if keep == 'none':
            # Remove all duplicates (keep no duplicated columns)
            mask_to_keep = ~self.df.columns.duplicated(keep=False)
        else:
            # Keep first or last occurrence
            mask_to_keep = ~self.df.columns.duplicated(keep=keep)
        
        if inplace:
            self.df = self.df.loc[:, mask_to_keep]
            result_df = None
        else:
            result_df = self.df.loc[:, mask_to_keep]
        
        removed_count = original_shape[1] - mask_to_keep.sum()
        print(f"✅ Removed {removed_count} duplicate columns (kept '{keep}' occurrence)")
        
        return result_df, {
            "action": "removed",
            "removed_columns": removed_count,
            "keep": keep,
            "new_shape": (original_shape[0], mask_to_keep.sum())
        }