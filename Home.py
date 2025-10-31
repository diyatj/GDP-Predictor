import streamlit as st

st.set_page_config(
    page_title="GDPredict", # Web app title
    page_icon="👋",
)

st.write("# Welcome to GDPredict! 👋")

st.sidebar.success("Select a demo above.")

from pathlib import Path

# Attempt to import the model helper module created from the notebook
try:
    import final_gdp
except Exception as e:
    final_gdp = None
    _final_gdp_import_error = e

## ---- GDP model integration ----
st.markdown("## GDP Model — Actual vs Predicted")

csv_path = Path(__file__).resolve().parent / "US GDP Factors(Sheet1).csv"

if not csv_path.exists():
    st.error(f"CSV file not found at: {csv_path}. Put the dataset in the app folder.")
else:
    if final_gdp is None:
        st.error("Could not import model utilities. Check that scikit-learn is installed.")
        with st.expander("Import error details"):
            st.write(str(_final_gdp_import_error))
    else:
        # Load available countries (cached to avoid re-reading the CSV on every small interaction)
        @st.cache_data
        def _get_countries(p):
            try:
                return final_gdp.get_countries(str(p))
            except Exception:
                return []

        countries = _get_countries(csv_path)
        options = ["All"] + countries if countries else ["All"]

        selected_country = st.selectbox("Select country (filters dataset)", options)

        # On selection, train/evaluate on the filtered dataset (or whole dataset if 'All')
        with st.spinner("Training / evaluating model for selection..."):
            country_arg = None if selected_country == "All" else selected_country
            try:
                result = final_gdp.train_and_eval(str(csv_path), country=country_arg)
            except Exception as e:
                st.error("Error training or evaluating the model. See details below.")
                st.exception(e)
                result = None

        if result is not None:
            # Metrics
            col1, col2 = st.columns(2)
            col1.metric("R²", f"{result['r2']:.3f}")
            col2.metric("RMSE", f"{result['rmse']:,.2f}")

            # Correlations display (between target and features for the selected data)
            if result.get("correlations") is not None:
                with st.expander("Correlations (target vs features)"):
                    corr = result["correlations"]
                    st.dataframe(corr.to_frame(name="correlation"))
                    # quick bar chart visualization
                    try:
                        st.bar_chart(corr)
                    except Exception:
                        pass

            # Coefficients
            with st.expander("Model coefficients"):
                st.dataframe(result["coef_df"])

            # Predictions chart
            try:
                pred_df = final_gdp.predictions_dataframe(result)
                st.line_chart(pred_df)
            except Exception:
                st.write("Couldn't build chart from predictions; showing raw arrays instead.")
                st.write({
                    "years": result["years_test"].tolist(),
                    "actual": result["y_test"].tolist(),
                    "predicted": result["y_pred"].tolist(),
                })

            caption = f"Data source: {csv_path.name}"
            if result.get("country"):
                caption += f" — filtered by {result.get('country_col')} = {result.get('country')}"
            st.caption(caption)