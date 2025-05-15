import pandas as pd
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
import streamlit as st

# ---------------------------- Price Calculation
def calculate_price_custom(distance, slab_rates, slab_distances):
    price = 0
    previous_boundary = 0
    for i, boundary in enumerate(slab_distances):
        slab_length = boundary - previous_boundary
        if distance <= boundary:
            if i == 0:
                price = slab_rates[0]
            else:
                price += slab_rates[i] * (distance - previous_boundary)
            return price
        else:
            if i == 0:
                price = slab_rates[0]
            else:
                price += slab_rates[i] * slab_length
        previous_boundary = boundary
    price += slab_rates[-1] * (distance - previous_boundary)
    return price

# ---------------------------- Objective Function
def total_absolute_deviation(slab_rates, slab_distances, data):
    return sum(
        abs(calculate_price_custom(row['Distance From Crusher'], slab_rates, slab_distances) - row['One_Way_Price'])
        for _, row in data.iterrows()
    )

# ---------------------------- Optimization
def fit_slab_rates(data, slab_distances, max_deviation):
    n_slabs = len(slab_distances) + 1
    initial_guess = [1000.0] + [10.0] * (n_slabs - 1)
    bounds = [(1e-5, None)] * n_slabs
    constraints = [{"type": "ineq", "fun": lambda r, i=i: r[i] - r[i + 1]} for i in range(1, n_slabs - 1)]
    constraints.append({"type": "ineq", "fun": lambda r: max_deviation - total_absolute_deviation(r, slab_distances, data)})

    result = minimize(
        lambda r: sum(
            (calculate_price_custom(row['Distance From Crusher'], r, slab_distances) - row['One_Way_Price']) ** 2
            for _, row in data.iterrows()
        ),
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )
    return result

# ---------------------------- Grid Search
def grid_search_slab_configurations(data, min_slabs=3, max_slabs=5, max_dev_pct=0.2):
    best_result = None
    best_config = None
    max_distance = data['Distance From Crusher'].max()
    max_dev = data['One_Way_Price'].mean() * max_dev_pct
    results = []

    for slab_count in range(min_slabs, max_slabs + 1):
        candidate_points = np.linspace(1, max_distance - 1, 12)
        for boundary_set in combinations(candidate_points, slab_count - 1):
            slab_distances = sorted(list(boundary_set))
            slab_distances.append(max_distance)
            result = fit_slab_rates(data, slab_distances[:-1], max_dev)
            if result.success:
                score = result.fun
                results.append((score, slab_distances, result.x))
                if best_result is None or score < best_result.fun:
                    best_result = result
                    best_config = slab_distances
    return best_config, best_result, results

# ---------------------------- Result Table Generator
def generate_result_df(company, quantity, slab_starts, slab_ends, slab_rates):
    rows = []
    for i, (start, end, rate) in enumerate(zip(slab_starts, slab_ends, slab_rates), start=1):
        if i == 1:
            one_way = f"₹{rate:.2f}"
            two_way = f"₹{2 * rate:.2f}"
        else:
            one_way = f"₹{rate:.2f}/km"
            two_way = f"₹{2 * rate:.2f}/km"
        rows.append({
            "Crusher Name": company,
            "Quantity_of_Material": quantity,
            "Slabs": f"Slab_{i}",
            "Slabs in KM": f"{int(start)} to {int(end)}",
            "One Way Price": one_way,
            "Two Way Price": two_way
        })
    return pd.DataFrame(rows)

# ---------------------------- Streamlit App
def app():
    st.title("🚛 Slab Rate Optimization for Crushers")

    uploaded_file = st.file_uploader("📤 Upload CSV file", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_cols = {'company_name', 'Distance From Crusher', 'logistics_value', 'quantity_value', 'created_time'}
        if required_cols.issubset(df.columns):
            df['One_Way_Price'] = df['logistics_value'] / 2.0

            company = st.selectbox("🏢 Select Crusher Name", df['company_name'].unique())
            company_df = df[df['company_name'] == company]

            quantity = st.selectbox("📦 Select Quantity of Material", company_df['quantity_value'].unique())
            filtered_df = company_df[company_df['quantity_value'] == quantity]

            filtered_df['created_time'] = pd.to_datetime(filtered_df['created_time'])
            available_dates = filtered_df['created_time'].dt.date.unique()

            # Date range selection
            selected_start_date, selected_end_date = st.date_input(
                "📅 Select Date Range",
                min_value=available_dates.min(),
                max_value=available_dates.max(),
                value=(available_dates.min(), available_dates.max())
            )

            # Filter the data based on the selected date range
            date_filtered_df = filtered_df[(filtered_df['created_time'].dt.date >= selected_start_date) &
                                           (filtered_df['created_time'].dt.date <= selected_end_date)]

            key_prefix = f"{company}_{quantity}_{selected_start_date}_{selected_end_date}"

            if st.button("🔍 Get Optimized Slab Rates"):
                if not date_filtered_df.empty:
                    best_config, best_result, _ = grid_search_slab_configurations(date_filtered_df)
                    if best_result:
                        slab_starts = [0] + best_config[:-1]
                        slab_ends = best_config
                        slab_rates = best_result.x

                        st.session_state[f"result_df_{key_prefix}"] = generate_result_df(
                            company, quantity, slab_starts, slab_ends, slab_rates
                        )
                        st.session_state["slab_rates"] = slab_rates
                        st.session_state["slab_distances"] = best_config[:-1]
                        st.success("✅ Optimized slab rates computed successfully!")
                else:
                    st.warning("⚠️ No data available for the selected combination.")

            result_key = f"result_df_{key_prefix}"
            if result_key in st.session_state:
                result_df = st.session_state[result_key]
                st.dataframe(result_df)

                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results as CSV", data=csv, file_name='optimized_slab_rates.csv', mime='text/csv')

                # ---------------------------- Price by Distance
                st.subheader("📈 Calculate Price by Distance")
                distance = st.slider("Select Distance (in KM)", min_value=0, max_value=100, value=10)

                # Parse slab table
                slabs = []
                for _, row in result_df.iterrows():
                    slab_range = row['Slabs in KM']
                    start_km, end_km = map(int, slab_range.split(" to "))
                    if "/km" in row['Two Way Price']:
                        rate = float(row['Two Way Price'].replace("₹", "").replace("/km", ""))
                        slabs.append({"start": start_km, "end": end_km, "rate": rate, "type": "per_km"})
                    else:
                        rate = float(row['Two Way Price'].replace("₹", ""))
                        slabs.append({"start": start_km, "end": end_km, "rate": rate, "type": "flat"})

                # Calculate price
                total_price = 0.0
                remaining_distance = distance
                price_details = []

                for slab in slabs:
                    slab_length = slab["end"] - slab["start"]
                    if remaining_distance <= 0:
                        break

                    if distance <= slab["end"]:
                        used_km = max(0, remaining_distance)
                        if slab["type"] == "flat":
                            total_price = slab["rate"]
                            price_details.append(f"📌 First {slab['end']} km at flat rate: ₹{slab['rate']:.2f}")
                        else:
                            total_price += used_km * slab["rate"]
                            price_details.append(f"📌 {used_km} km within slab {slab['start']}–{slab['end']} km at ₹{slab['rate']}/km: ₹{used_km * slab['rate']:.2f}")
                        break
                    else:
                        if slab["type"] == "flat":
                            total_price = slab["rate"]
                            price_details.append(f"📌 First {slab['end']} km at flat rate: ₹{slab['rate']:.2f}")
                        else:
                            total_price += slab_length * slab["rate"]
                            price_details.append(f"📌 {slab_length} km within slab {slab['start']}–{slab['end']} km at ₹{slab['rate']}/km: ₹{slab_length * slab['rate']:.2f}")
                        remaining_distance -= slab_length

                st.info(f"🛣️ **Selected Distance**: {distance} km")
                st.success(f"💸 **Two Way Price**: ₹{total_price:.2f}")

                with st.expander("🔍 Pricing Breakdown"):
                    for detail in price_details:
                        st.write(detail)
        else:
            st.error("❌ Uploaded file missing required columns.")
    else:
        st.info("📥 Please upload a CSV file to begin.")

# ---------------------------- Run Streamlit App
if __name__ == '__main__':
    app()
