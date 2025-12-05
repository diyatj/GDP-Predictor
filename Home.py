import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="GDPredict",  # Web app title
    page_icon="👋",
    layout="wide",  # Use wide layout for sidebar
)

# Setup sidebar
st.sidebar.title("⚙️ Simulation Settings")

st.write("# GDPredict")


# Path to default CSV
csv_path = Path(__file__).resolve().parent / "datasets" / "US_GDP.csv"
effective_csv_path = csv_path  # will be overridden if registry chooses a different file

# 🧭 Debug snippet: list all CSVs the app can see
folder = Path(__file__).resolve().parent / "datasets"
# for p in folder.glob("*.csv"):
#     st.write("-", p.name)

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

if not csv_path.exists():
    st.error(f"CSV file not found at: {csv_path}. Put the dataset in the app folder.")
else:
    if final_gdp is None:
        st.error("Could not import model utilities. Check that scikit-learn is installed.")
        with st.expander("Import error details"):
            st.write(str(_final_gdp_import_error))
    else:
        # Prefer a registry of per-country CSV files if available.
        @st.cache_data
        def _get_registry_countries():
            try:
                return country_registry.list_countries() if country_registry else []
            except Exception:
                return []

        registry_countries = _get_registry_countries()

        # show a placeholder so the app doesn't run the model on first load
        PLACEHOLDER = "Select a country"

        result = None  # will hold model result

        if registry_countries:
            options = [PLACEHOLDER] + registry_countries
            selected_country = st.selectbox("Select a country", options)

            # If a CSV is registered for the country, use it. Otherwise fall back to default CSV
            csv_for_country = None if selected_country == PLACEHOLDER else country_registry.get_csv_for_country(selected_country)
            effective_csv = csv_for_country if csv_for_country is not None else csv_path
            effective_csv_path = Path(effective_csv) if effective_csv is not None else csv_path

            # If nothing selected yet, show a blank chart and an instruction
            if selected_country == PLACEHOLDER:
                # st.info("Select a country from the dropdown to run the model and show results.")
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
                # st.info("Select a country from the dropdown to run the model and show results.")
                empty_df = pd.DataFrame({"Actual": [], "Predicted": []})
                st.line_chart(empty_df)
                result = None
                effective_csv_path = csv_path
            else:
                with st.spinner("Training / evaluating model for selection..."):
                    country_arg = selected_country
                    try:
                        result = final_gdp.train_and_eval(str(csv_path), country=country_arg)
                        effective_csv_path = csv_path
                    except Exception as e:
                        st.error("Error training or evaluating the model. See details below.")
                        st.exception(e)
                        result = None
                        effective_csv_path = csv_path

        # ----- AFTER selection & training: show graph first, then metrics, then details -----
        if result is not None:
            # ---- Build predictions & forecast + plot FIRST ----
            try:
                pred_df = final_gdp.predictions_dataframe(result)
            except Exception:
                pred_df = None

            st.sidebar.subheader("Forecast Settings")
            method = st.sidebar.selectbox("Forecast method", ["trend", "constant"], index=0)
            n_quarters = st.sidebar.slider("Forecast quarters", min_value=1, max_value=20, value=8)

            # Event shock controls in sidebar
            st.sidebar.subheader("Shock Factors")
            
            #st.sidebar.caption(
            #   "Feature growth uses the average pct change of the last 3 observations:\n"
            #    "pct_change_t = (x_t - x_{t-1}) / x_{t-1}; "
            #    "g = mean of the last 3 pct_change values; "
            #    "projection: x_next = x_current * (1 + g)."
            #)

            # Load events for selected country
            events_csv = None
            if registry_countries and selected_country != PLACEHOLDER:
                try:
                    events_csv = country_registry.get_events_csv_for_country(selected_country)
                except Exception:
                    events_csv = None
            
            events_df = pd.DataFrame()
            if events_csv and events_csv.exists():
                try:
                    events_df = final_gdp.load_events(str(events_csv))
                except Exception:
                    events_df = pd.DataFrame()
            
            # Display event checkboxes in sidebar
            selected_events = {}
            if not events_df.empty:
                for _, row in events_df.iterrows():
                    event_name = row.get("event", "unknown")
                    selected_events[event_name] = st.sidebar.checkbox(
                        f"{event_name.replace('_', ' ').title()}",
                        value=False,
                        help=f"GDP impact: {row.get('gdp_impact', 'N/A')} | Growth shock: {row.get('growth_shock', 'N/A')}%",
                    )
            
            # Get quarter index for any selected event
            shock_year_index = 0
            selected_event_name = None
            last_year = result.get("last_year", None)
            
            if any(selected_events.values()) and last_year is not None:
                # At least one event is selected; ask which quarter to apply it
                forecast_quarters = []
                for i in range(1, n_quarters + 1):
                    year = int(last_year) + (i // 4)
                    quarter = (i % 4) if (i % 4) != 0 else 4
                    forecast_quarters.append(f"{year} Q{quarter}")
                chosen_quarter = st.sidebar.selectbox(
                    "Shock quarter",
                    forecast_quarters,
                    index=0,
                )
                shock_year_index = forecast_quarters.index(chosen_quarter)
                # Get the first selected event
                selected_event_name = next(k for k, v in selected_events.items() if v)
            elif any(selected_events.values()):
                shock_year_index = 0

            # Build forecast DataFrame
            try:
                fut_df = final_gdp.forecast_next_quarters(
                    result,
                    n_quarters=n_quarters,
                    method=method,
                    shock_quarter_index=shock_year_index,
                )
                # rename forecast column to avoid collision with test 'Predicted'
                fut_df = fut_df.rename(columns={"Predicted": "Forecast"})
                
                # Apply selected event shock if any
                if selected_event_name and not events_df.empty:
                    event_row = events_df[events_df["event"] == selected_event_name]
                    if not event_row.empty:
                        forecast_values = fut_df["Forecast"].values
                        forecast_values = final_gdp.apply_event_shock(
                            forecast_values, 
                            event_row.iloc[0], 
                            shock_year_index,
                            growth_rates=result.get("growth_rates"),
                            baseline_gdp=result.get("y_test")[-1] if len(result.get("y_test", [])) > 0 else None
                        )
                        fut_df["Forecast"] = forecast_values

                # 🔗 Make forecast dates continue exactly from last Predicted date
                try:
                    # Ensure datetime index
                    if pred_df is not None and not pred_df.empty:
                        pred_df.index = pd.to_datetime(pred_df.index)
                        last_pred_date = pred_df.index.max()
                    else:
                        last_pred_date = None

                    fut_df.index = pd.to_datetime(fut_df.index)

                    if last_pred_date is not None and not fut_df.empty:
                        # Infer the historical frequency (quarterly for US_GDP.csv)
                        freq = pd.infer_freq(pred_df.index) or "QS"

                        # First forecast point is next period after last_pred_date
                        first_forecast_date = last_pred_date + pd.tseries.frequencies.to_offset(freq)

                        # Rebuild forecast index so it follows immediately after the red line
                        # But keep the first point at last_pred_date for connection
                        fut_df_dates = pd.date_range(
                            start=first_forecast_date,
                            periods=len(fut_df) - 1,  # -1 because we'll prepend the connection point
                            freq=freq,
                        )
                        fut_df.index = pd.Index([last_pred_date] + fut_df_dates.tolist())
                except Exception:
                    pass
                
                # Connect forecast to prediction by setting the first forecast value to last predicted value
                if pred_df is not None and not pred_df.empty and fut_df is not None and not fut_df.empty:
                    last_pred_val = pred_df.iloc[-1]["Predicted"]
                    fut_df.iloc[0, fut_df.columns.get_loc("Forecast")] = last_pred_val

            except Exception:
                fut_df = None

            # Combine and plot
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
                    combined_idx = [
                        int(str(i)[:4]) if str(i).isdigit() or str(i)[:4].isdigit() else None
                        for i in combined.index.astype(str)
                    ]
                    # build a DataFrame column for sorting where None values go last
                    sort_df = pd.DataFrame(
                        {"year_sort": [v if v is not None else 10**9 for v in combined_idx]},
                        index=combined.index,
                    )
                    combined = (
                        combined.assign(_sort=sort_df["year_sort"])
                        .sort_values("_sort")
                        .drop(columns=["_sort"])
                    )
                except Exception:
                    pass

                # Plot combined results with Plotly for clearer legends and styling
                try:
                    fig = go.Figure()

                    # x values (years) as strings
                    x = combined.index.astype(str).tolist()

                    if "Actual" in combined.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=combined["Actual"],
                                name="Actual",
                                mode="lines+markers",
                                line=dict(width=2),
                            )
                        )
                    if "Predicted" in combined.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=combined["Predicted"],
                                name="Predicted (test)",
                                mode="lines+markers",
                                line=dict(width=2),
                            )
                        )
                    
                    if "Forecast" in combined.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=combined["Forecast"],
                                name="Forecast (next quarters)",
                                mode="lines+markers",
                                line=dict(width=2, dash="dash"),
                            )
                        )
                    # Determine units based on country
                    if "United States" in selected_country or "US" in selected_country:
                        y_label = "GDP (Billions of US Dollars)"
                    elif "Japan" in selected_country:
                        y_label = "GDP (Billions of Yen)"
                    elif "Israel" in selected_country:
                        y_label = "GDP (Millions of New Israeli Shekels)"
                    else:
                        y_label = "GDP"

                    fig.update_layout(
                        title={
                            "text": f"{selected_country} GDP",
                            "font": {"size": 28},   # make title bigger    # center title
                             },           
                        xaxis_title="Year",
                        yaxis_title=y_label,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1,
                        ),
                        template="plotly_white",
                    )

                    # 👇 GRAPH SHOWS FIRST (after selection)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    # fallback to Streamlit chart if Plotly fails
                    st.line_chart(combined)

                with st.expander("Prediction & Forecast numbers"):
                    st.dataframe(combined)

            # ---- THEN show metrics (R² and Average Prediction Error) ----
            col1, col2 = st.columns(2)
            col1.metric("R²", f"{result['r2']:.3f}")

            # Use RMSE percentage instead of raw RMSE
            pct = result.get("rmse_pct_mean", None)
            if pct is not None and not pd.isna(pct):
                col2.metric("Average Prediction Error (%)", f"{pct:.2f}%")
            else:
                col2.metric("Average Prediction Error (%)", "N/A")

            # ---- THEN the rest: correlations, coefficients, etc. ----
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

            # Show the CSV actually used (registry may have supplied a different file)
            try:
                csv_used_name = effective_csv_path.name
            except Exception:
                csv_used_name = csv_path.name

            caption = f"Data source: {csv_used_name}"
            if result.get("country"):
                caption += f" — filtered by {result.get('country_col')} = {result.get('country')}"
            st.caption(caption)
