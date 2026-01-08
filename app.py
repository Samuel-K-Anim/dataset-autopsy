import streamlit as st
import pandas as pd
import plotly.express as px
import io
from autopsy_engine import DataCoroner, DataHealer, generate_impact_report
from report_generator import generate_pdf_report

# --- CONFIG ---
st.set_page_config(page_title="Dataset Autopsy Lab", page_icon="🩺", layout="wide", initial_sidebar_state='expanded')

# Temporary storage for uploaded and modified data
if 'df_main' not in st.session_state:
    st.session_state.df_main = None # The working copy
if 'df_original' not in st.session_state:
    st.session_state.df_original = None # The original uploaded copy
if 'change_log' not in st.session_state:
    st.session_state.change_log = [] # Log of changes made


st.title("🩺 Dataset Autopsy Lab")

st.markdown("""
**The Data Coroner is in.** Upload your dataset to detect missing values, outliers, skewness, and leakage risks.
""")

# --- SIDEBAR: UPLOAD ---
with st.sidebar:
    st.header("⚙️ Controls")
    mode = st.radio("Select Mode", ["👁️ Preview Mode", "✏️ Editing Mode"])
    
    st.divider()
    st.header("📂 Evidence Locker")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    

# --- DATA LOADING ---
if uploaded_file:
    # Load data only if not already loaded
    if st.session_state.df_main is None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Save to session state
            st.session_state.df_main = df
            st.session_state.df_original = df.copy()
            st.session_state.change_log = [] # Reset logs
            st.rerun() # Refresh to show data
        except Exception as e:
            st.error(f"Error loading file: {e}")

# --- MAIN INTERFACE ---
if st.session_state.df_main is not None:
    df = st.session_state.df_main # Current working data
    coroner = DataCoroner(df) # Initialize Data Coroner

    # === MODE 1: PREVIEW (Read Only) ===
    if mode == "👁️ Preview Mode":
        st.info("You are in **Preview Mode**. Switch to **Editing Mode** to apply fixes.")
                # --- TABBED REPORT ---
        tab1, tab2, tab3, tab4 = st.tabs(["🫀 Vitals", "⚠️ Abnormalities", "🩸 Leakage & Imbalance", "📝 Recommendations"])
        
        # === TAB 1: VITALS (Overview) ===
        with tab1:
            stats = coroner.get_vital_signs()
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", stats["rows"])
            c2.metric("Columns", stats["cols"])
            c3.metric("Missing Cells", stats["missing_cells"])
            c4.metric("Duplicates", stats["duplicates"])
            
            st.divider()
            
            c_mem, c_num, c_cat = st.columns(3)
            c_mem.metric("Memory Usage", f"{stats['memory_mb']} MB")
            c_num.metric("Numeric Features", stats["numeric_count"])
            c_cat.metric("Categorical Features", stats["categorical_count"])

            st.subheader("👀 Sample Data")
            st.dataframe(df.head(), width='stretch')

        # === TAB 2: ABNORMALITIES (Data Quality Issues) ===
        with tab2:
            st.subheader("⚠️ Critical Data Quality Issues")

            # 1. CONSTANT FEATURES
            constants = coroner.check_constant_features()
            if not constants.empty:
                st.error("💤 Constant Features Detected (Zero Variance)")
                st.dataframe(constants, width='stretch')
            
            # 2. TYPE MISMATCHES
            mismatches = coroner.check_type_mismatches()
            if mismatches:
                st.warning(f"🔢 Type Mismatch: These columns look like numbers but are stored as text: {mismatches}")

            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("📈 Skewness")
                skew_df = coroner.check_skewness()

                # Show only if significant skewness found
                if not skew_df.empty:
                    st.dataframe(skew_df, width='stretch')
                    st.warning("High skewness can confuse models. Consider Log Transformation.")
                    
                    # Visualize top skewed feature
                    top_skew = skew_df.iloc[0]["Column"]
                    fig = px.histogram(df, x=top_skew, title=f"Distribution of {top_skew}")
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.success("No significant skewness detected.")
            
            with c2:
                st.subheader("📉 Outlier (IQR)")
                outlier_df = coroner.check_outliers_iqr()

                # Show only if outliers found
                if not outlier_df.empty:
                    st.dataframe(outlier_df, width='stretch')
                    st.error("Outliers detected. Check for data entry errors.")
                else:
                    st.success("No outliers detected.")
            
            st.divider()
            st.subheader("🩸 Missing Values Analysis")
            missing_df = coroner.check_missing()

            # Show only if missing values exist
            if not missing_df.empty:
                st.dataframe(missing_df, width='stretch')
                # Visualize missing data
                st.bar_chart(missing_df.set_index("Missing Count")["Percentage"])
            else:
                st.success("Dataset is fully populated (0% Missing).")

        # === TAB 3: LEAKAGE & IMBALANCE ===
        with tab3:
            st.subheader("🔗 Correlation Heatmap")
            
            # Correlation Matrix for Numeric Features
            numeric_df = df.select_dtypes(include=['number'])
            
            if not numeric_df.empty:
                corr_matrix = numeric_df.corr()
                fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
                st.plotly_chart(fig_corr, width='stretch')
            
            st.divider()

            st.subheader("🕵️‍♂️ Target Analysis")
            
            # Auto select Target Column
            likely_target = coroner.detect_likely_target()
            target = st.selectbox("Select Target Column", df.columns, index=df.columns.get_loc(likely_target))

            # 1. LEAKAGE CHECK
            if target:
                leakage = coroner.check_leakage(target)
                if not leakage.empty:
                    st.error("🚨 LEAKAGE DETECTED! (Correlation > 95%)")
                    st.dataframe(leakage)
                else:
                    st.success(f"No leakage found for target: {target}.")
                
                st.divider()
                
                # 2. CLASS IMBALANCE CHECK
                st.write("### ⚖️ Class Balance Check")
                imbalance = coroner.check_class_balance(target)
                if imbalance is not None:
                    st.warning("⚠️ Imbalance Detected in Target Class")
                    st.dataframe(imbalance, width='stretch')
                    st.bar_chart(df[target].value_counts())
                else:
                    st.info("Target is balanced or continuous (Regression).")

        # === TAB 4: RECOMMENDATIONS (Next Steps) ===
        with tab4:
            st.header("Autopsy Report")
            rec_data = coroner.generate_recommendations()
            # Display ML Readiness Score with color coding
            score = rec_data["score"]
            if score > 80:
                color = "green"
                msg = "Excellent! Ready for ML."
            elif score > 50:
                color = "orange"
                msg = "Needs Preprocessing."
            else:
                color = "red"
                msg = "Critical Issues Found."
                
            st.markdown(f"##### ML Readiness Score: :{color}[{score}/100]")
            st.progress(score / 100)
            st.caption(msg)
            st.divider()
            for rec in rec_data["recommendations"]: st.info(rec)
            
    # === MODE 2: EDITING (Apply Fixes) ===
    elif mode == "✏️ Editing Mode":
        st.warning("⚠️ You are in **Editing Mode**. Changes affect your temporary copy, not the uploaded file.")
        
        edit_tab1, edit_tab2, edit_tab3 = st.tabs(["📝 Rename Columns", "🧪 Treatment Room", "📥 Export & Report"])
        
        # --- SUB-TAB: RENAME COLUMNS ---
        with edit_tab1:
            st.subheader("Rename Columns")
            st.write("Edit column names directly below:")
            
            # Dynamic inputs for each column
            new_names = {}
            cols = st.columns(3)
            for i, col in enumerate(df.columns):
                # Distribute inputs across 3 columns
                with cols[i % 3]:
                    new_names[col] = st.text_input(f"Rename '{col}'", value=col, key=f"rename_{col}")
            
            if st.button("Apply Renames"):
                # Detect changes
                changed = {k: v for k, v in new_names.items() if k != v}
                if changed:
                    healer = DataHealer(df)
                    st.session_state.df_main = healer.rename_columns(changed)
                    st.session_state.change_log.append(f"Renamed {len(changed)} columns.")
                    st.success("Columns renamed!")
                    st.rerun()
                else:
                    st.info("No changes detected.")

        # -- SUB-TAB: TREATMENT ROOM ---
        with edit_tab2:
            st.subheader("Apply Treatments")
            
            # Missing Value Handling
            with st.expander("🩸 Handle Missing Values", expanded=True):
                miss_strat = st.selectbox(
                    "Choose Imputation Strategy:", 
                    ["Median (Robust to Outliers)", "Mean (Average)", "Zero (Fill 0)","Drop"]
                )
                if st.button("Apply Fill"):
                    healer = DataHealer(df)
                    # Extract key from selection
                    strat_key = miss_strat.split(" ")[0] 
                    st.session_state.df_main = healer.fill_missing(strategy=strat_key)
                    if strat_key != "Drop":
                        st.session_state.change_log.append(f"Filled missing values using {strat_key}.")
                    else:
                        st.session_state.change_log.append(f"You dropped missing values using {strat_key}.")
                    st.rerun()
            
            # 1. Drop Columns
            with st.expander("🗑️ Drop Columns"):
                st.info("Select columns to remove (e.g., IDs, Usernames, Duplicate Features).")
                cols_to_drop = st.multiselect("Select columns to drop:", df.columns)
                if st.button("Drop Selected Columns"):
                    healer = DataHealer(df)
                    if cols_to_drop:
                        st.session_state.df_main = healer.drop_specific_columns(cols_to_drop)
                        st.session_state.change_log.append(f"Dropped {len(cols_to_drop)} columns: {', '.join(cols_to_drop)}")
                        st.rerun()
            
            # 2. Drop Rows
            with st.expander("🗑️ Drop Rows"):
                st.info("Select specific rows by their Index ID to remove.")
                # Using df.index to let user select indices
                rows_to_drop = st.multiselect("Select Row Indices to drop:", df.index)
                if st.button("Drop Selected Rows"):
                    healer = DataHealer(df)
                    if rows_to_drop:
                        st.session_state.df_main = healer.drop_specific_rows(rows_to_drop)
                        st.session_state.change_log.append(f"Dropped {len(rows_to_drop)} rows.")
                        st.rerun()

            # 3. Outlier Handling
            with st.expander("📉 Handle Outliers"):
                st.write("Outliers are detected using the IQR method.")
                iqr_factor = st.slider("Sensitivity (Lower = More Strict)", 1.5, 3.0, 1.5, 0.5)
                
                if st.button("Cap Outliers"):
                    healer = DataHealer(df)
                    st.session_state.df_main = healer.cap_outliers_iqr(factor=iqr_factor)
                    st.session_state.change_log.append(f"Capped outliers (Factor: {iqr_factor}).")
                    st.rerun()

            #  4. Categorical Encoding
            with st.expander("🔠 Categorical Encoding"):
                st.info("Converts text columns into numbers for ML models.")
                enc_strat = st.radio("Strategy", ["Auto (Recommended)", "One-Hot (All)", "Label (All)"])
                if st.button("Apply Encoding"):
                    healer = DataHealer(df)
                    strat_key = enc_strat.split(" ")[0]
                    new_df, encoded_list = healer.encode_categorical(strategy=strat_key)
                    st.session_state.df_main = new_df
                    st.session_state.change_log.append(f"Encoded columns: {', '.join(encoded_list)}")
                    st.rerun()

            #  5. Date Conversion
            with st.expander("📅 Date Conversion"):
                # Auto-detect object columns that look like dates
                potential_dates = df.select_dtypes(include=['object']).columns.tolist()
                date_cols = st.multiselect("Select columns to convert to DateTime:", potential_dates)
                if st.button("Convert Dates"):
                    healer = DataHealer(df)
                    if date_cols:
                        st.session_state.df_main = healer.convert_to_datetime(date_cols)
                        st.session_state.change_log.append(f"Converted dates: {', '.join(date_cols)}")
                        st.rerun()

            # 6. General Cleanup
            with st.expander("🧹 General Cleanup"):
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("👯 Remove Duplicates"):
                        healer = DataHealer(df)
                        st.session_state.df_main = healer.drop_duplicates()
                        st.session_state.change_log.append("Removed duplicates.")
                        st.rerun()
                with c2:
                    if st.button("📈 Fix Skewness (Log)"):
                        healer = DataHealer(df)
                        st.session_state.df_main = healer.log_transform_skewed()
                        st.session_state.change_log.append("Log-transformed skewed features.")
                        st.rerun()
            
            st.divider()
            st.write("### Current Change Log:")
            if st.session_state.change_log:
                for log in st.session_state.change_log:
                    st.caption(f"✅ {log}")
            else:
                st.caption("No changes applied yet.")

        # -- SUB-TAB: EXPORT & REPORT ---
        with edit_tab3:
            st.subheader("Download & Report")
            
            # Generate Impact Summary
            impact = generate_impact_report(st.session_state.df_original, st.session_state.df_main)
            
            # 1. Download Cleaned Data
            st.write("### 📥 Download Cleaned Dataset")
            try:
                csv = st.session_state.df_main.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Cleaned CSV", csv, f"cleaned_{uploaded_file.name}", "text/csv")
            except Exception as e:
                st.error(f"Error generating CSV: {e}")

            # 2. Generate PDF Report
            st.write("---")
            st.write("Generate a professional report detailing your edits and the data's health.")
            
            if st.button("📄 Generate Professional PDF Report"):
                # Re-run final diagnostics on cleaned data
                final_stats = coroner.get_vital_signs()
                final_missing = coroner.check_missing()
                final_outliers = coroner.check_outliers_iqr()
                final_skew = coroner.check_skewness()
                final_rec = coroner.generate_recommendations()
                
                pdf_data = generate_pdf_report(
                    uploaded_file.name,
                    final_stats,
                    final_missing,
                    final_outliers,
                    final_skew,
                    final_rec["recommendations"],
                    final_rec["score"],
                    change_log=st.session_state.change_log,
                    impact_summary=impact
                )
                
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_data,
                    file_name="Impact_Report.pdf",
                    mime="application/pdf"
                )
else:
    st.info("👈Please upload a dataset to begin the autopsy.")
