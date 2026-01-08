import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import LabelEncoder

class DataCoroner:
    def __init__(self, df):
        self.df = df
        # Identify numeric columns automatically
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- 1. VITAL SIGNS ---
    def get_vital_signs(self):
        """
        A. Dataset Overview (Autopsy Summary)
        Returns: Dict of basic stats
        """
        buffer = self.df.memory_usage(deep=True).sum() / 1024**2 # MB
        missing_count = self.df.isnull().sum().sum()
        
        return {
            "rows": self.df.shape[0],
            "cols": self.df.shape[1],
            "duplicates": self.df.duplicated().sum(),
            "missing_cells": missing_count,
            "missing_percent": round((missing_count / self.df.size) * 100, 2),
            "numeric_count": len(self.numeric_cols),
            "categorical_count": len(self.categorical_cols),
            "memory_mb": round(buffer, 2)
        }

    # --- 2. TARGET DETECTION ---
    def detect_likely_target(self):
        """Heuristic to guess the target column."""
        # Priority 1: Common names
        common_names = ['target', 'label', 'class', 'outcome', 'price', 'churn', 'survived', 'salary']
        for col in self.df.columns:
            if col.lower() in common_names:
                return col
        # Priority 2: Last column (common convention)
        return self.df.columns[-1]

    # --- 3. CONSTANT FEATURES ---
    def check_constant_features(self):
        """Finds columns with 0 variance or >95% same value."""
        constant_data = []
        for col in self.df.columns:
            # 1. Check constant (1 unique value)
            if self.df[col].nunique() <= 1:
                constant_data.append({"Column": col, "Issue": "Constant (1 value)", "Risk": "Useless 🔴"})
            # 2. Check near-constant (>95% same value)
            else:
                top_freq = self.df[col].value_counts(normalize=True).iloc[0]
                if top_freq > 0.95:
                    constant_data.append({"Column": col, "Issue": f"Near-Constant ({top_freq:.1%})", "Risk": "Noise 🟡"})
        return pd.DataFrame(constant_data)

    # --- 4. TYPE MISMATCHES ---
    def check_type_mismatches(self):
        """Checks if Object columns are actually numbers (e.g. '$100', '1,000')."""
        mismatches = []
        for col in self.categorical_cols:
            # Attempt to convert to numeric
            numeric_conversion = pd.to_numeric(self.df[col], errors='coerce')
            success_rate = numeric_conversion.notna().mean()
            
            # If >80% conversion success, flag it
            if success_rate > 0.8:
                mismatches.append(col)
        return mismatches

    # --- 5. CLASS IMBALANCE ---
    def check_class_balance(self, target_col):
        """Checks if a categorical target is imbalanced."""
        if target_col in self.numeric_cols and self.df[target_col].nunique() > 20:
            return None # Not categorical
            
        counts = self.df[target_col].value_counts(normalize=True)
        # Flag if any class <10%
        if counts.min() < 0.10:
            return pd.DataFrame({
                "Class": counts.index, 
                "Percentage": (counts.values * 100).round(1),
                "Status": ["Minority 🔴" if x < 0.1 else "Majority" for x in counts.values]
            })
        return None

    # --- 6. DETAILED ANALYSIS ---
    def check_missing(self):
        """
        B. Missing Values Analysis
        Returns: DataFrame of columns with missing values and %
        """
        missing = self.df.isnull().sum()
        missing = missing[missing > 0] # Filter only columns with missing
        
        if missing.empty:
            return pd.DataFrame()
            
        missing_df = pd.DataFrame({
            "Missing Count": missing,
            "Percentage": (missing / len(self.df) * 100).round(2)
        })
        # Add Severity
        missing_df["Severity"] = np.where(missing_df["Percentage"] > 30, "Critical 🔴", "Manageable 🟡")
        return missing_df.sort_values(by="Percentage", ascending=False)

    # --- 7. ADVANCED ANALYSES ---
    def check_skewness(self):
        """
        D. Skewness & Distribution
        Returns: DataFrame of highly skewed columns (>1 or <-1)
        """
        skew_data = []
        for col in self.numeric_cols:
            # Calculate skewness, ignoring NaNs
            clean_col = self.df[col].dropna()
            if len(clean_col) > 0:
                s_val = skew(clean_col)
                if abs(s_val) > 1: # Highly skewed
                    skew_data.append({
                        "Column": col,
                        "Skewness": round(s_val, 2),
                        "Verdict": "Positively Skewed (Tail Right)" if s_val > 0 else "Negatively Skewed (Tail Left)"
                    })
        return pd.DataFrame(skew_data)

    # --- 8. OUTLIER DETECTION ---
    def check_outliers_iqr(self):
        """
        C. Outlier Detection (IQR Method)
        Returns: DataFrame summary of outliers per column
        """
        outlier_data = []
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            count = len(outliers)
            
            if count > 0:
                outlier_data.append({
                    "Column": col,
                    "Outlier Count": count,
                    "Percent Affected": round((count / len(self.df)) * 100, 2)
                })
        return pd.DataFrame(outlier_data)

    # --- 9. LEAKAGE DETECTION ---
    def check_leakage(self, target_col):
        """
        F. Leakage Detection (The Advanced Feature)
        Checks correlation > 95% with target.
        """
        if target_col not in self.numeric_cols:
            return pd.DataFrame()
            
        # Calculate absolute correlations with target
        correlations = self.df[self.numeric_cols].corrwith(self.df[target_col]).abs()
        
        # Find features with correlation > 0.95 (but not 1.0)
        suspects = correlations[(correlations > 0.95) & (correlations < 1.0)]
        
        if suspects.empty:
            return pd.DataFrame()
            
        return pd.DataFrame({
            "Suspect Feature": suspects.index,
            "Correlation": suspects.values.round(4),
            "Risk": "EXTREME 🚨"
        })

    # --- 10. RECOMMENDATIONS & SCORING ---
    def generate_recommendations(self):
        """
        Generates actionable 'Next Steps' and an ML Readiness Score.
        """
        recommendations = []
        score = 100
        
        # 1. Missing Data Impact
        missing = self.check_missing()
        if not missing.empty:
            count = len(missing)
            score -= (count * 2) # Deduct 2 points per column with missing
            recommendations.append(f"🩸 **Missing Data:** {count} columns have missing values. Impute with Median (for skewed) or Mean (for normal).")
        
        # 2. Outliers Impact
        outliers = self.check_outliers_iqr()
        if not outliers.empty:
            score -= 10
            recommendations.append(f"📉 **Outliers:** Detected in {len(outliers)} columns. Consider Robust Scaling or Winsorization.")
            
        # 3. Skewness Impact
        skewness = self.check_skewness()
        if not skewness.empty:
            score -= 5
            recommendations.append(f"📈 **Skewness:** Highly skewed distributions found. Apply Log Transformation (np.log1p) to normalize.")
            
        # 4. Encoding Suggestions
        if self.categorical_cols:
            recommendations.append(f"🔠 **Encoding:** {len(self.categorical_cols)} categorical columns found. Check Treatment Room for One-Hot/Label Encoding.")
            
        # 5. Date Handling Suggestions
        for col in self.categorical_cols:
            if any(x in col.lower() for x in ['date', 'time', 'year', 'day']):
                recommendations.append(f"📅 **Date Handling:** Column '{col}' looks like a date. Convert it in the Treatment Room.")

        # 6. ID Column Detection
        for col in self.df.columns:
            # Simple heuristic: Column name contains 'id' and unique values equal to row count
            if "id" in col.lower() and self.df[col].nunique() == len(self.df):
                recommendations.append(f"🆔 **ID Column:** '{col}' appears to be a unique identifier. Drop this before training.")

        # Ensure score is not negative
        score = max(0, score)
        
        return {
            "score": score,
            "recommendations": recommendations
        }


class DataHealer:
    def __init__(self, df):
        self.df = df.copy() # Work on a copy to avoid modifying original directly
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- 1. RENAME COLUMNS ---
    # Accepts a dictionary {old_name: new_name}
    def rename_columns(self, rename_dict):
        """Renames columns based on a dictionary"""
        self.df = self.df.rename(columns=rename_dict)
        return self.df

    # --- 2. HANDLE MISSING VALUES ---
    # Strategy options: "Median", "Mean", "Zero", "Drop"
    def fill_missing(self, strategy="Median"):
        """Fills missing values based on user choice."""
        for col in self.numeric_cols:
            if strategy == "Median":
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif strategy == "Mean":
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif strategy == "Zero":
                self.df[col] = self.df[col].fillna(0)
            elif strategy == "Drop":
                self.df[col] = self.df[col].dropna()
                
        # Categorical columns - fill with mode
        for col in self.categorical_cols:
            if not self.df[col].mode().empty:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            else:
                self.df[col] = self.df[col].fillna("Unknown")
        return self.df

    # --- 3. OUTLIER TREATMENT ---
    def cap_outliers_iqr(self, factor=1.5):
        """Caps outliers using a variable IQR factor."""
        for col in self.numeric_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower = Q1 - (factor * IQR)
            upper = Q3 + (factor * IQR)

            self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
        return self.df

    # --- 4. SKEWNESS CORRECTION ---
    def log_transform_skewed(self):
        """Applies Log(x+1) to highly skewed columns"""
        for col in self.numeric_cols:
            # Calculate skewness
            if (self.df[col].dropna() >= 0).all():
                skewness = skew(self.df[col].dropna())
                skewness = skew(self.df[col].dropna())

                if abs(skewness) > 1:
                    self.df[col] = np.log1p(self.df[col])
        return self.df
    
    # --- 5. DROP DUPLICATES ---
    def drop_duplicates(self):
        self.df = self.df.drop_duplicates()
        return self.df

    # --- 6. DATE CONVERSION & FEATURE EXTRACTION ---
    def convert_to_datetime(self, cols):
        """Converts selected columns to datetime objects."""
        for col in cols:
            self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
            # Extract features
            self.df[f"{col}_year"] = self.df[col].dt.year
            self.df[f"{col}_month"] = self.df[col].dt.month
            self.df[f"{col}_day"] = self.df[col].dt.day
            # Drop original column
            self.df = self.df.drop(columns=[col])
        return self.df

    # --- 7. ENCODE CATEGORICAL VARIABLES ---
    def encode_categorical(self, strategy="Auto"):
        """Encodes categorical columns based on cardinality."""
        le = LabelEncoder()
        
        # Identify categorical columns
        cat_cols = self.df.select_dtypes(exclude=[np.number, 'datetime']).columns.tolist()
        encoded_cols = []
                
        for col in cat_cols:
            if self.df[col].notna().sum() == 0:
                continue  # Skip columns with all NaNs

            unique_count = self.df[col].nunique()
            
            # Strategy: One-Hot Encoding
            if strategy == "One-Hot" or (strategy == "Auto" and unique_count < 10):
                self.df = pd.get_dummies(self.df, columns=[col], prefix=col, drop_first=True)
                encoded_cols.append(f"One-Hot: {col}")
                
            # Strategy: Label Encoding
            elif strategy == "Label" or (strategy == "Auto" and unique_count >= 10):
                # Convert to string to avoid issues
                self.df[col] = self.df[col].astype(str) 
                self.df[col] = le.fit_transform(self.df[col])
                encoded_cols.append(f"Label: {col}")
                
        return self.df, encoded_cols

    # --- 8. DROP SPECIFIC COLUMNS ---
    def drop_specific_columns(self, cols_to_drop):
        """Drops a list of columns from the dataframe."""
        self.df = self.df.drop(columns=cols_to_drop, errors='ignore')
        return self.df

    # --- 9. DROP SPECIFIC ROWS ---
    def drop_specific_rows(self, indices_to_drop):
        """Drops list of rows by index"""
        self.df = self.df.drop(index=indices_to_drop, errors='ignore')
        return self.df


def generate_impact_report(df_original, df_final):
    """Compares the before and after state to quantify improvements."""
    report = []
    
    # Check Rows
    rows_diff = df_original.shape[0] - df_final.shape[0]
    if rows_diff > 0:
        report.append(f"Removed {rows_diff} duplicate/unwanted rows.")
        
    # Check Missing Values
    missing_old = df_original.isnull().sum().sum()
    missing_new = df_final.isnull().sum().sum()
    if missing_old > missing_new:
        report.append(f"Filled {missing_old - missing_new} missing values.")
        
    # Renamed/Modified Columns
    old_cols = set(df_original.columns)
    new_cols = set(df_final.columns)
    renamed = old_cols - new_cols
    if renamed:
        report.append(f"Renamed/Modified {len(renamed)} columns: {', '.join(list(renamed)[:3])}...")
        
    if not report:
        report.append("No significant structural changes detected.")
        
    return report