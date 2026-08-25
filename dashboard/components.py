"""Streamlit rendering components for the pricing dashboard.

All business/pricing logic is delegated to dashboard/pricing.py (pure
helpers) and, transitively, to the existing src/pricing modules. This
module only lays out widgets.

The pricing-breakdown component (`pricing_breakdown_panel`) and the
"Why this price?" / "Model Information" expanders are shared between
Page 1 (Project Pricing) and Page 2 (Price Simulator) -- there is exactly
one explanation system, used for both a real project apartment and a
custom simulator apartment.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.pricing import (
    format_currency,
    format_percent,
    format_value,
    summarize_historical_transactions,
)
from src.config.settings import (
    CURRENT_MARKET_DATA_TYPE,
    CURRENT_MARKET_WEIGHT,
    HISTORICAL_MARKET_WEIGHT,
    PROJECT_ADDRESS,
    PROJECT_CITY,
    PROJECT_NAME,
    PROJECT_NEIGHBORHOOD,
    TARGET_MARKET_SEGMENT,
)

APARTMENT_TABLE_COLUMNS = {
    "apartment_id": "Apartment",
    "rooms": "Rooms",
    "interior_area_sqm": "Area (sqm)",
    "floor_min": "Floor",
    "property_type": "Type",
    "parking_count": "Parking",
    "balcony_area_sqm": "Balcony (sqm)",
    "final_strategy_price": "Final Price",
    "final_strategy_price_per_sqm": "Price / sqm",
}

CHARACTERISTIC_FIELDS = [
    ("apartment_id", "Apartment ID", ""),
    ("rooms", "Rooms", ""),
    ("floor_min", "Floor (min)", ""),
    ("floor_max", "Floor (max)", ""),
    ("num_levels", "Levels", ""),
    ("interior_area_sqm", "Interior Area", " sqm"),
    ("balcony_area_sqm", "Balcony Area", " sqm"),
    ("balcony_direction", "Balcony Direction", ""),
    ("directions", "Directions", ""),
    ("parking_count", "Parking", ""),
    ("storage_area_sqm", "Storage", " sqm"),
    ("garden_area_sqm", "Garden", " sqm"),
    ("roof_area_sqm", "Roof", " sqm"),
    ("is_top_floor", "Top Floor", ""),
    ("property_type", "Property Type", ""),
]


def project_header() -> None:
    st.title(PROJECT_NAME)
    subtitle_parts = [p for p in [PROJECT_ADDRESS, PROJECT_NEIGHBORHOOD, PROJECT_CITY] if p]
    st.caption(" · ".join(subtitle_parts))


def kpi_cards(kpis: dict) -> None:
    cols = st.columns(6)
    cols[0].metric("Apartments", format_value(kpis.get("apartment_count")))
    cols[1].metric("Avg. Final Price", format_currency(kpis.get("average_final_price"), compact=True))
    cols[2].metric(
        "Avg. Price / sqm", format_currency(kpis.get("average_final_price_per_sqm"))
    )
    cols[3].metric("Min. Final Price", format_currency(kpis.get("min_final_price"), compact=True))
    cols[4].metric("Max. Final Price", format_currency(kpis.get("max_final_price"), compact=True))
    cols[5].metric("Total Project Value", format_currency(kpis.get("total_project_value"), compact=True))


def weights_strategy_summary() -> None:
    from src.config.settings import (
        COMPANY_POSITIONING_ADJUSTMENT_PCT,
        INVENTORY_STRATEGY_ADJUSTMENT_PCT,
        SALES_PHASE_ADJUSTMENT_PCT,
    )

    st.markdown("**Pricing Configuration**")
    cols = st.columns(5)
    cols[0].metric("Historical Market Weight", format_percent(HISTORICAL_MARKET_WEIGHT).lstrip("+"))
    cols[1].metric("Current Market Weight", format_percent(CURRENT_MARKET_WEIGHT).lstrip("+"))
    cols[2].metric("Company Positioning", format_percent(COMPANY_POSITIONING_ADJUSTMENT_PCT))
    cols[3].metric("Sales Phase", format_percent(SALES_PHASE_ADJUSTMENT_PCT))
    cols[4].metric("Inventory Strategy", format_percent(INVENTORY_STRATEGY_ADJUSTMENT_PCT))


def apartment_table(pricing_df: pd.DataFrame) -> None:
    display_df = pricing_df[list(APARTMENT_TABLE_COLUMNS.keys())].rename(
        columns=APARTMENT_TABLE_COLUMNS
    )
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Final Price": st.column_config.NumberColumn(format="₪%d"),
            "Price / sqm": st.column_config.NumberColumn(format="₪%d"),
            "Area (sqm)": st.column_config.NumberColumn(format="%.1f"),
            "Balcony (sqm)": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def apartment_selector(pricing_df: pd.DataFrame) -> int:
    apartment_ids = sorted(pricing_df["apartment_id"].tolist())
    return st.selectbox(
        "Select an apartment to view its pricing breakdown",
        options=apartment_ids,
        format_func=lambda i: f"Apartment {i}",
    )


def characteristics_panel(row: dict) -> None:
    st.markdown("**Apartment Characteristics**")
    cols = st.columns(3)
    for i, (field, label, suffix) in enumerate(CHARACTERISTIC_FIELDS):
        cols[i % 3].write(f"{label}: **{format_value(row.get(field), suffix)}**")


def pricing_breakdown_panel(breakdown: dict, area_sqm: float | None = None) -> None:
    """The shared pricing-breakdown component, used identically on Page 1
    (a real project apartment) and Page 2 (a custom simulator apartment).

    All numbers are read directly from `breakdown` (either a stored
    apartment_pricing_recommendations.csv row or a freshly computed
    result from src/pricing/custom_apartment_pricing.py) -- no rounded
    intermediate value is invented here.
    """
    hist_price = breakdown.get("historical_base_price")
    market_price = breakdown.get("current_market_price")
    hist_weight = breakdown.get("historical_weight", HISTORICAL_MARKET_WEIGHT)
    market_weight = breakdown.get("current_market_weight", CURRENT_MARKET_WEIGHT)
    hist_contribution = breakdown.get("historical_contribution")
    market_contribution = breakdown.get("current_market_contribution")
    recommended = breakdown.get("recommended_marketing_price")
    final_price = breakdown.get("final_strategy_price")
    final_price_per_sqm = breakdown.get("final_strategy_price_per_sqm")

    st.markdown("**Market Signals**")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Historical Model", format_currency(hist_price))
        st.caption(
            f"{format_currency(hist_price)} × {format_percent(hist_weight).lstrip('+')} "
            f"= {format_currency(hist_contribution)}"
        )
    with c2:
        st.metric("Current Market Model", format_currency(market_price))
        st.caption(
            f"{format_currency(market_price)} × {format_percent(market_weight).lstrip('+')} "
            f"= {format_currency(market_contribution)}"
        )

    st.markdown("---")
    st.metric("Market Recommendation", format_currency(recommended))

    st.markdown("**Company Strategy Adjustments**")
    s1, s2, s3, s4 = st.columns(4)
    s1.write(f"Positioning\n\n**{format_percent(breakdown.get('company_positioning_adjustment_pct'))}**")
    s2.write(f"Sales Phase\n\n**{format_percent(breakdown.get('sales_phase_adjustment_pct'))}**")
    s3.write(f"Inventory\n\n**{format_percent(breakdown.get('inventory_strategy_adjustment_pct'))}**")
    manual_pct = breakdown.get("manual_adjustment_pct")
    manual_amount = breakdown.get("manual_adjustment_amount")
    manual_label = format_percent(manual_pct)
    if manual_amount:
        manual_label += f" / {format_currency(manual_amount)}"
    s4.write(f"Manual\n\n**{manual_label}**")

    st.markdown("---")
    f1, f2 = st.columns(2)
    f1.metric("Final Strategy Price", format_currency(final_price))
    f2.metric("Final Price / sqm", format_currency(final_price_per_sqm))


def why_this_price_expander() -> None:
    with st.expander("Why this price?"):
        st.markdown(
            f"""
**Historical Signal**

The Historical Model is trained on completed residential transactions
retrieved through the GovMap / Israeli Tax Authority pipeline. Each
transaction's price is adjusted to current terms using the CBS Housing
Price Index before the model is trained. The historical model uses only
the features that actually exist reliably in that transaction data:
**interior area, rooms, and floor**. It does not use parking, storage,
balcony, or direction, because GovMap transaction records do not
reliably include them.

**Current Market Signal — {CURRENT_MARKET_DATA_TYPE.replace('_', ' ').title()}**

The Current Market Model is trained on **Synthetic POC Current Market
Data** — a manually supplied dataset that mimics the shape of data we
would eventually receive from a source such as Yad2, Madlan, or a
developer/project feed. It is **not** real listing data and must not be
interpreted as one. It exists to demonstrate the full pricing
architecture until a real current-market data source is connected. This
model uses the richer apartment characteristics available in that
dataset: area, rooms, floor, balcony area, parking, storage, garden,
roof, balcony direction, property type, market segment
(**{TARGET_MARKET_SEGMENT}** for this project), and top-floor status.

**Market Recommendation**

The two independent signals are blended using the currently configured
weights: **{format_percent(HISTORICAL_MARKET_WEIGHT).lstrip('+')} Historical + {format_percent(CURRENT_MARKET_WEIGHT).lstrip('+')} Current Market**.

**Company Strategy**

The market recommendation can then be adjusted by explicit company
strategy parameters — overall positioning, sales phase, inventory
strategy, and an optional apartment-specific manual adjustment. These are
separate business decisions layered on top of the property valuation
models, not another valuation model themselves.
            """
        )


def model_information_expander(hist_report: dict | None, market_report: dict | None) -> None:
    with st.expander("Model Information"):
        st.markdown("**Historical Model**")
        if hist_report is None:
            st.warning("Historical model report is unavailable.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", format_currency(hist_report.get("mae")))
            c2.metric("RMSE", format_currency(hist_report.get("rmse")))
            c3.metric("R²", f"{hist_report.get('r2', 0):.4f}")
            st.caption(f"Features: {', '.join(hist_report.get('training_features', []))}")

        st.markdown("---")
        st.markdown("**Synthetic Current Market Model Metrics**")
        st.info(
            "These metrics are calculated on synthetic POC market data and should "
            "not be interpreted as real-world model accuracy."
        )
        if market_report is None:
            st.warning("Current Market model report is unavailable.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", format_currency(market_report.get("mae")))
            c2.metric("RMSE", format_currency(market_report.get("rmse")))
            c3.metric("R²", f"{market_report.get('r2', 0):.4f}")
            st.caption(f"Features: {', '.join(market_report.get('features', []))}")


def supporting_data_expander(
    transactions_df: pd.DataFrame | None, market_df: pd.DataFrame | None
) -> None:
    with st.expander("View Supporting Data"):
        tab1, tab2 = st.tabs(["Historical Transactions", "Current Market"])

        with tab1:
            st.caption("Historical transactions used to build the historical pricing signal.")
            if transactions_df is None or transactions_df.empty:
                st.warning("Historical transactions data is unavailable.")
            else:
                summary = summarize_historical_transactions(transactions_df)

                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Total", summary["total"])
                m2.metric("Eligible", summary["eligible"])
                m3.metric("Excluded", summary["excluded"])
                m4.metric("CBS Enriched", summary["cbs_enriched"])
                m5.metric("CBS Missing", summary["cbs_missing"])
                m6.metric("Used for Model", summary["used_for_historical_model"])
                st.caption(
                    "\"Used for Model\" applies the strict residential whitelist "
                    "(property_type=\"דירה\" and deal_nature in {\"דירה בבית קומות\", \"דירת גן\"}) "
                    "plus sold-fraction/full-ownership normalization -- see \"Why this price?\" "
                    "for details. A transaction can be eligible for general market-data purposes "
                    "yet still not be used to train the Historical Regression Model."
                )

                filter_choice = st.radio(
                    "Filter",
                    ["All", "Used for Historical Model"],
                    horizontal=True,
                    key="tx_filter_choice",
                )
                display_df = transactions_df
                if filter_choice == "Used for Historical Model":
                    # canonical boolean -- the same field
                    # src.pricing.regression_features.select_training_transactions
                    # uses to build the actual training set. Deliberately NOT
                    # is_eligible_comparable, which is a broader, general
                    # market-data-quality flag (see summarize_historical_transactions).
                    if "used_for_historical_model" in display_df.columns:
                        display_df = display_df[display_df["used_for_historical_model"] == True]  # noqa: E712
                    else:
                        st.warning(
                            "used_for_historical_model is not available in the loaded "
                            "transactions data -- showing all rows instead."
                        )

                columns_to_show = [
                    c
                    for c in [
                        "address",
                        "transaction_date",
                        "rooms",
                        "area_sqm",
                        "floor",
                        "original_price",
                        "sold_fraction",
                        "full_ownership_price",
                        "full_ownership_price_per_sqm",
                        "adjusted_price",
                        "adjusted_price_per_sqm",
                        "property_type",
                        "deal_nature",
                        "is_eligible_comparable",
                        "exclusion_reason",
                        "suspicious_partial_transaction",
                        "used_for_historical_model",
                        "historical_model_exclusion_reason",
                        "source",
                    ]
                    if c in display_df.columns
                ]
                st.dataframe(
                    display_df[columns_to_show].head(200), use_container_width=True, hide_index=True
                )

        with tab2:
            st.caption("Synthetic POC Current Market Data — not Yad2 or Madlan data.")
            if market_df is None or market_df.empty:
                st.warning("Current Market data is unavailable.")
            else:
                st.metric("Total Listings", len(market_df))
                columns_to_show = [
                    c
                    for c in [
                        "rooms",
                        "area_sqm",
                        "floor",
                        "asking_price",
                        "price_per_sqm",
                        "parking_count",
                        "storage_area_sqm",
                        "balcony_area_sqm",
                        "balcony_direction",
                        "garden_area_sqm",
                        "roof_area_sqm",
                        "property_type",
                        "market_segment",
                    ]
                    if c in market_df.columns
                ]
                st.dataframe(
                    market_df[columns_to_show].head(200), use_container_width=True, hide_index=True
                )


def simulator_inputs_form(ranges: dict, property_types: list, balcony_directions: list) -> dict | None:
    """Renders the custom-apartment input form. Returns the input dict
    only when the user submits (clicks Calculate Price); otherwise None."""
    with st.form("simulator_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            rooms = st.number_input(
                "Rooms", min_value=1, max_value=10, value=int(ranges["rooms"]["default"]), step=1
            )
            interior_area_sqm = st.number_input(
                "Interior Area (sqm)",
                min_value=10.0,
                max_value=500.0,
                value=float(ranges["interior_area_sqm"]["default"]),
                step=1.0,
            )
            floor = st.number_input(
                "Floor", min_value=0, max_value=50, value=int(ranges["floor"]["default"]), step=1
            )
            num_levels = st.number_input(
                "Levels",
                min_value=1,
                max_value=3,
                value=int(ranges["num_levels"]["default"]),
                step=1,
                help="Descriptive only — not used as a model predictor.",
            )
        with c2:
            balcony_area_sqm = st.number_input(
                "Balcony Area (sqm)",
                min_value=0.0,
                max_value=200.0,
                value=float(ranges["balcony_area_sqm"]["default"]),
                step=1.0,
            )
            balcony_direction = st.selectbox("Balcony Direction", options=balcony_directions)
            property_type = st.selectbox("Property Type", options=property_types)
            is_top_floor = st.checkbox("Top Floor", value=False)
        with c3:
            parking_count = st.number_input(
                "Parking Spots",
                min_value=0,
                max_value=5,
                value=int(ranges["parking_count"]["default"]),
                step=1,
            )
            storage_area_sqm = st.number_input(
                "Storage Area (sqm)",
                min_value=0.0,
                max_value=50.0,
                value=float(ranges["storage_area_sqm"]["default"]),
                step=1.0,
            )
            garden_area_sqm = st.number_input(
                "Garden Area (sqm)",
                min_value=0.0,
                max_value=200.0,
                value=float(ranges["garden_area_sqm"]["default"]),
                step=1.0,
            )
            roof_area_sqm = st.number_input(
                "Roof Area (sqm)",
                min_value=0.0,
                max_value=200.0,
                value=float(ranges["roof_area_sqm"]["default"]),
                step=1.0,
            )

        submitted = st.form_submit_button("Calculate Price", type="primary")

    if not submitted:
        return None

    return {
        "rooms": rooms,
        "interior_area_sqm": interior_area_sqm,
        "floor": floor,
        "num_levels": num_levels,
        "balcony_area_sqm": balcony_area_sqm,
        "balcony_direction": balcony_direction,
        "parking_count": parking_count,
        "storage_area_sqm": storage_area_sqm,
        "garden_area_sqm": garden_area_sqm,
        "roof_area_sqm": roof_area_sqm,
        "is_top_floor": is_top_floor,
        "property_type": property_type,
    }


def scenario_inputs() -> dict:
    st.markdown("**Scenario Adjustments**")
    st.caption("Scenario only — project configuration is not modified.")
    c1, c2, c3 = st.columns(3)
    with c1:
        company_positioning_pct = st.slider(
            "Company Positioning Adjustment %", min_value=-10.0, max_value=10.0, value=0.0, step=0.5
        ) / 100.0
    with c2:
        sales_phase_pct = st.slider(
            "Sales Phase Adjustment %", min_value=-10.0, max_value=10.0, value=0.0, step=0.5
        ) / 100.0
    with c3:
        inventory_strategy_pct = st.slider(
            "Inventory Strategy Adjustment %", min_value=-10.0, max_value=10.0, value=0.0, step=0.5
        ) / 100.0

    with st.expander("Apartment-specific manual adjustment (optional)"):
        m1, m2 = st.columns(2)
        with m1:
            manual_adjustment_pct = (
                st.number_input(
                    "Manual Adjustment %", min_value=-20.0, max_value=20.0, value=0.0, step=0.5
                )
                / 100.0
            )
        with m2:
            manual_adjustment_amount = st.number_input(
                "Manual Adjustment Amount (₪)", min_value=-500_000, max_value=500_000, value=0, step=1000
            )

    return {
        "company_positioning_pct": company_positioning_pct,
        "sales_phase_pct": sales_phase_pct,
        "inventory_strategy_pct": inventory_strategy_pct,
        "manual_adjustment_pct": manual_adjustment_pct,
        "manual_adjustment_amount": float(manual_adjustment_amount),
    }
