import streamlit as st

st.set_page_config(
    page_title="GDPredict", # Web app title
    page_icon="assets/GDPredict Logo.svg",    # Web app icon
)

st.write("# GDPredict")

st.sidebar.success("Select a demo above.")

from pathlib import Path
import pandas as pd
import plotly.graph_objects as go


csv_path = Path(__file__).resolve().parent / "datasets" / "US_GDP.csv"

# 🧭 Debug snippet: list all CSVs the app can see
folder = Path(__file__).resolve().parent / "datasets"
st.write("CSV files detected in 'datasets' folder:")
for p in folder.glob("*.csv"):
    st.write("-", p.name)

if not csv_path.exists():
    st.error(f"CSV file not found at: {csv_path}. Put the dataset in the app folder.")
    
# Attempt to import the model helper module created from the notebook
try:
    import final_gdp
except Exception as e:
    final_gdp = None
    _final_gdp_import_error = e
try:
    import country_registry
except Exception:
    country_registry = None

## ---- GDP model integration ----
st.markdown("## GDP Model — Actual vs Predicted")

csv_path = Path(__file__).resolve().parent / "datasets" / "US_GDP.csv"

if not csv_path.exists():
    st.error(f"CSV file not found at: {csv_path}. Put the dataset in the app folder.")
else:
    if final_gdp is None:
        st.error("Could not import model utilities. Check that scikit-learn is installed.")
        with st.expander("Import error details"):
            st.write(str(_final_gdp_import_error))
    else:
        # Prefer a registry of per-country CSV files if available. Fallback to reading the
        # CSV and extracting country values from a column (the notebook-style flow).
        @st.cache_data
        def _get_registry_countries():
            try:
                return country_registry.list_countries() if country_registry else []
            except Exception:
                return []

        registry_countries = _get_registry_countries()

        # show a placeholder so the app doesn't run the model on first load
        PLACEHOLDER = "-- Select a country --"
        if registry_countries:
            options = [PLACEHOLDER] + registry_countries
            selected_country = st.selectbox("Select a country", options)

            # If a CSV is registered for the country, use it. Otherwise fall back to default CSV
            csv_for_country = None if selected_country == PLACEHOLDER else country_registry.get_csv_for_country(selected_country)
            effective_csv = csv_for_country if csv_for_country is not None else csv_path
            effective_csv_path = Path(effective_csv) if effective_csv is not None else csv_path

            # If nothing selected yet, show a blank chart and an instruction
            if selected_country == PLACEHOLDER:
                st.info("Select a country from the dropdown to run the model and show results.")
                empty_df = pd.DataFrame({"Actual": [], "Predicted": []})
                st.line_chart(empty_df)
                result = None
                effective_csv_path = csv_path
            else:
                with st.spinner("Training / evaluating model for selection..."):
                    try:
                        # If using per-country CSVs, don't pass a country filter to train_and_eval.
                        result = final_gdp.train_and_eval(str(effective_csv), country=None)
                    except Exception as e:
                        st.error("Error training or evaluating the model. See details below.")
                        st.exception(e)
                        result = None
        else:
            # No registry entries; fall back to the CSV containing a country column and filter rows by value
            @st.cache_data
            def _get_countries_from_csv(p):
                try:
                    return final_gdp.get_countries(str(p))
                except Exception:
                    return []

            countries = _get_countries_from_csv(csv_path)
            options = [PLACEHOLDER] + countries if countries else [PLACEHOLDER]
            selected_country = st.selectbox("Select country (filters dataset)", options)

            if selected_country == PLACEHOLDER:
                st.info("Select a country from the dropdown to run the model and show results.")
                empty_df = pd.DataFrame({"Actual": [], "Predicted": []})
                st.line_chart(empty_df)
                result = None
                effective_csv_path = csv_path
            else:
                with st.spinner("Training / evaluating model for selection..."):
                    country_arg = selected_country
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

            # Combine Predictions (test window) and 5-year Forecast into a single chart
            try:
                pred_df = final_gdp.predictions_dataframe(result)
            except Exception:
                pred_df = None

            # Forecast controls
            method = st.selectbox("Forecast method", ["trend", "constant"], index=0)
            n_years = st.slider("Forecast years", min_value=1, max_value=10, value=5)

            try:
                fut_df = final_gdp.forecast_next_years(result, n_years=n_years, method=method)
                # rename forecast column to avoid collision with test 'Predicted'
                fut_df = fut_df.rename(columns={"Predicted": "Forecast"})
            except Exception as e:
                fut_df = None

            if pred_df is None and fut_df is None:
                st.write("Couldn't build predictions or forecast.")
            else:
                # Build a combined DataFrame indexed by Year with columns: Actual, Predicted, Forecast
                parts = []
                if pred_df is not None:
                    parts.append(pred_df)
                if fut_df is not None:
                    parts.append(fut_df)

                combined = pd.concat(parts, axis=0)
                # Ensure index ordering by converting Year to string and sorting chronologically where possible
                try:
                    # try to parse index as int years for sorting
                    combined_idx = [int(str(i)[:4]) if str(i).isdigit() or str(i)[:4].isdigit() else None for i in combined.index.astype(str)]
                    # build a DataFrame column for sorting where None values go last
                    sort_df = pd.DataFrame({"year_sort": [v if v is not None else 10**9 for v in combined_idx]}, index=combined.index)
                    combined = combined.assign(_sort=sort_df["year_sort"]).sort_values("_sort").drop(columns=["_sort"])
                except Exception:
                    pass

                # Plot combined results with Plotly for clearer legends and styling
                try:
                    fig = go.Figure()

                    # x values (years) as strings
                    x = combined.index.astype(str).tolist()

                    if "Actual" in combined.columns:
                        fig.add_trace(go.Scatter(x=x, y=combined["Actual"], name="Actual", mode="lines+markers", line=dict(width=2)))
                    if "Predicted" in combined.columns:
                        fig.add_trace(go.Scatter(x=x, y=combined["Predicted"], name="Predicted (test)", mode="lines+markers", line=dict(width=2)))
                    if "Forecast" in combined.columns:
                        fig.add_trace(go.Scatter(x=x, y=combined["Forecast"], name="Forecast (next years)", mode="lines+markers", line=dict(width=2, dash="dash")))

                    fig.update_layout(
                        title="GDP — Actual / Predicted / Forecast",
                        xaxis_title="Year",
                        yaxis_title="GDP",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        template="plotly_white",
                    )

                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    # fallback to Streamlit chart if Plotly fails
                    st.line_chart(combined)

                with st.expander("Prediction & Forecast numbers"):
                    st.dataframe(combined)

            # Show the CSV actually used (registry may have supplied a different file)
            try:
                csv_used_name = effective_csv_path.name
            except Exception:
                csv_used_name = csv_path.name
            caption = f"Data source: {csv_used_name}"
            if result.get("country"):
                caption += f" — filtered by {result.get('country_col')} = {result.get('country')}"
            st.caption(caption)