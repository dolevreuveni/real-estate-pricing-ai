"""Real Estate Pricing dashboard entrypoint.

Run:
    streamlit run app.py

Two main pages only: Project Pricing and Price Simulator (see
dashboard/components.py). This file wires pages together; all
presentation logic lives in dashboard/, all pricing/business logic
lives in src/pricing and src/data (never duplicated here).
"""
import streamlit as st

from dashboard import components, data
from dashboard.pricing import (
    compute_project_kpis,
    derive_simulator_input_ranges,
    get_apartment_detail,
    get_valid_categories,
    simulate_custom_apartment,
)
from src.pricing.custom_apartment_pricing import CUSTOM_APARTMENT_FIELDS  # noqa: F401  (documents inputs)

st.set_page_config(page_title="Real Estate Pricing", layout="wide")


def page_project_pricing() -> None:
    components.project_header()

    pricing_df = data.load_pricing_recommendations()
    if pricing_df is None or pricing_df.empty:
        st.error(
            "Pricing recommendations are not available. Run "
            "scripts/train_baseline_pricing_model.py and then "
            "scripts/generate_pricing_recommendations.py to generate them."
        )
        return

    kpis = compute_project_kpis(pricing_df)
    components.kpi_cards(kpis)
    st.markdown("")
    components.weights_strategy_summary()
    st.markdown("---")

    st.subheader("Apartments")
    components.apartment_table(pricing_df)

    st.markdown("---")
    apartment_id = components.apartment_selector(pricing_df)
    row = get_apartment_detail(pricing_df, apartment_id)

    if row is None:
        st.warning(f"Apartment {apartment_id} was not found in the pricing table.")
        return

    st.subheader(f"Apartment {apartment_id} — Pricing Breakdown")
    components.characteristics_panel(row)
    st.markdown("")
    components.pricing_breakdown_panel(row, area_sqm=row.get("interior_area_sqm"))

    components.why_this_price_expander()

    hist_report = data.load_regression_model_report()
    market_report = data.load_current_market_model_report()
    components.model_information_expander(hist_report, market_report)

    transactions_df = data.load_transactions()
    market_df = data.load_current_market_raw()
    components.supporting_data_expander(transactions_df, market_df)


def page_price_simulator() -> None:
    components.project_header()
    st.subheader("Price Simulator")
    st.caption(
        "Create a custom apartment and get a pricing recommendation from the same "
        "pricing architecture used for the 39 project apartments."
    )

    apartments_df = data.load_apartments()
    market_df = data.load_current_market_raw()

    if market_df is None or market_df.empty:
        st.error("Current Market data is unavailable — the simulator cannot run without it.")
        return

    property_types = get_valid_categories(market_df, "property_type")
    balcony_directions = get_valid_categories(market_df, "balcony_direction")
    if not property_types or not balcony_directions:
        st.error("Could not derive valid property types / balcony directions from the Current Market data.")
        return

    ranges = derive_simulator_input_ranges(apartments_df, market_df)

    submitted_inputs = components.simulator_inputs_form(ranges, property_types, balcony_directions)
    # The scenario sliders below live outside the form, so adjusting one
    # alone reruns the script with the form's submit button not clicked
    # this time (submitted_inputs is None then). Persist the last
    # submitted apartment in session_state (this session only -- reset on
    # a fresh browser session, never written to disk) so scenario changes
    # recompute live without needing to resubmit the apartment form.
    if submitted_inputs is not None:
        st.session_state["simulator_apartment"] = submitted_inputs

    apartment_inputs = st.session_state.get("simulator_apartment")
    if apartment_inputs is None:
        return

    scenario = components.scenario_inputs()

    try:
        with st.spinner("Pricing custom apartment..."):
            historical_fit = data.get_historical_model()
            market_fit = data.get_current_market_model()
            breakdown = simulate_custom_apartment(
                apartment_inputs,
                historical_fit["model"],
                market_fit["model"],
                **scenario,
            )
    except FileNotFoundError as exc:
        st.error(f"A required data file is missing: {exc}")
        return
    except ValueError as exc:
        st.error(f"Could not price this apartment: {exc}")
        return
    except Exception:
        st.error("An unexpected error occurred while pricing this apartment. Please adjust the inputs and try again.")
        return

    st.markdown("---")
    st.subheader("Custom Apartment — Pricing Breakdown")
    components.characteristics_panel(apartment_inputs)
    st.markdown("")
    components.pricing_breakdown_panel(breakdown, area_sqm=apartment_inputs.get("interior_area_sqm"))

    components.why_this_price_expander()


def main() -> None:
    page = st.sidebar.radio("Navigation", ["Project Pricing", "Price Simulator"])

    if page == "Project Pricing":
        page_project_pricing()
    else:
        page_price_simulator()


if __name__ == "__main__":
    main()
