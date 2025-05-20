import pandas as pd
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
import streamlit as st
import time
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

# ---------------------------- Price Calculation
def calculate_price_custom(distance, slab_rates, slab_distances):
    distance = np.clip(distance, 0, slab_distances[-1])
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
    # Extra distance beyond the last slab
    if distance > slab_distances[-1]:
        price += (distance - slab_distances[-1]) * slab_rates[-1]
    return price

# ---------------------------- Optimization with Constraints
def fit_slab_rates(data, slab_distances, max_deviation):
    n_slabs = len(slab_distances) + 1
    initial_guess = [100.0] + [20.0, 15.0, 10.0, 5.0][:n_slabs - 1]
    bounds = [(1e-2, 1e4)] * n_slabs

    distances = data['Distance From Crusher'].values
    actual_prices = data['One_Way_Price'].values

    def objective_function(r):
        return np.sum([ 
            (calculate_price_custom(d, r, slab_distances) - p) ** 2
            for d, p in zip(distances, actual_prices)
        ])

    constraints = []
    for i in range(1, n_slabs - 1):
        constraints.append({'type': 'ineq', 'fun': lambda r, i=i: r[i] - r[i + 1] - 1e-3})

    result = minimize(
        objective_function,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={'disp': False, 'maxiter': 10000}
    )

    return result

# ---------------------------- Grid Search
@st.cache_data(show_spinner=False)
def grid_search_slab_configurations(data, min_slabs=3, max_slabs=5, max_dev_pct=0.2):
    best_result = None
    best_config = None
    best_score = float('inf')
    max_distance = data['Distance From Crusher'].max()
    max_dev = data['One_Way_Price'].mean() * max_dev_pct

    candidate_points = np.linspace(1, max_distance - 1, 6)

    for slab_count in range(min_slabs, max_slabs + 1):
        combinations_list = list(combinations(candidate_points, slab_count - 1))
        for slab_set in combinations_list:
            slab_distances = sorted(list(slab_set))
            slab_distances.append(max_distance)

            result = fit_slab_rates(data, slab_distances[:-1], max_dev)
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_result = result.x
                best_config = slab_distances

    return best_config, best_result

# ---------------------------- RMSE Calculation
def calculate_rmse(actual_prices, predicted_prices):
    return np.sqrt(mean_squared_error(actual_prices, predicted_prices))

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
    st.set_page_config(page_title="Slab Rate Optimizer", layout="centered")
    st.title("🚛 Slab Rate Optimization for Crushers")

    uploaded_file = st.file_uploader("📤 Upload CSV file", type=['csv'])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        required_cols = {'company_name', 'Distance From Crusher', 'logistics_value', 'quantity_value'}

        if not required_cols.issubset(df.columns):
            st.error("❌ Uploaded file missing required columns.")
            return

        df['One_Way_Price'] = df['logistics_value'] / 2.0

        company = st.selectbox("🏢 Select Crusher Name", df['company_name'].unique())
        quantity = st.selectbox("📦 Select Quantity", df[df['company_name'] == company]['quantity_value'].unique())
        filtered_df = df[(df['company_name'] == company) & (df['quantity_value'] == quantity)]

        key_prefix = f"{company}_{quantity}"

        slab_count = st.selectbox("🔢 Select Slab Count", options=[3, 4, 5], index=1)

        if st.button("🔍 Get Optimized Slab Rates"):
            if not filtered_df.empty:
                with st.spinner("⏳ Optimizing slab rates... Please wait"):
                    start_time = time.time()
                    best_config, best_result = grid_search_slab_configurations(filtered_df, min_slabs=slab_count, max_slabs=slab_count)
                    duration = time.time() - start_time
                    minutes, seconds = divmod(duration, 60)

                    if best_result is not None:
                        slab_starts = [0] + best_config[:-1]
                        slab_ends = best_config
                        slab_rates = best_result

                        result_df = generate_result_df(company, quantity, slab_starts, slab_ends, slab_rates)
                        st.session_state[f"result_df_{key_prefix}"] = result_df
                        st.session_state[f"slab_data_{key_prefix}"] = (slab_starts, slab_ends, slab_rates)
                        st.success(f"✅ Completed in {int(minutes)} min {seconds:.2f} sec.")

                        # Calculate and display RMSE
                        predicted_prices = [calculate_price_custom(d, slab_rates, slab_ends[:-1]) for d in filtered_df['Distance From Crusher']]
                        rmse = calculate_rmse(filtered_df['One_Way_Price'], predicted_prices)
                        st.subheader(f"📉 RMSE: {rmse:.2f}")

                    else:
                        st.warning("⚠️ Optimization failed.")
            else:
                st.warning("⚠️ No data available for selected filter.")

        # Make sure that the slab data exists before accessing slab_ends
        if f"slab_data_{key_prefix}" in st.session_state:
            slab_starts, slab_ends, slab_rates = st.session_state[f"slab_data_{key_prefix}"]

            result_key = f"result_df_{key_prefix}"
            result_df = st.session_state[result_key]

            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", data=csv, file_name="optimized_slab_rates.csv", mime="text/csv")

            st.subheader("📈 Estimate Price for Distance")
            distance = st.slider("Select Distance (in KM)", min_value=0, max_value=200, value=10)

            slabs = []
            for _, row in result_df.iterrows():
                start_km, end_km = map(int, row["Slabs in KM"].split(" to "))
                rate_text = row["Two Way Price"]
                if "/km" in rate_text:
                    rate = float(rate_text.replace("₹", "").replace("/km", "")) / 2  # Convert two-way to one-way
                    slabs.append({"start": start_km, "end": end_km, "rate": rate, "type": "per_km"})
                else:
                    rate = float(rate_text.replace("₹", "")) / 2  # Convert two-way to one-way
                    slabs.append({"start": start_km, "end": end_km, "rate": rate, "type": "flat"})

            total_price = 0.0
            remaining_distance = distance
            breakdown = []

            for slab in slabs:
                slab_length = slab["end"] - slab["start"]
                if remaining_distance <= 0:
                    break
                if distance <= slab["end"]:
                    used_km = max(0, remaining_distance)
                    if slab["type"] == "flat":
                        total_price = slab["rate"] * 2  # Two-way price for flat rates
                        breakdown.append(f"📌 Flat rate up to {slab['end']} km: ₹{total_price:.2f}")
                    else:
                        total_price += used_km * slab["rate"] * 2  # Two-way price for per km rates
                        breakdown.append(f"📌 {used_km} km × ₹{slab['rate'] * 2}/km = ₹{used_km * slab['rate'] * 2:.2f}")
                    break
                else:
                    if slab["type"] == "flat":
                        total_price = slab["rate"] * 2  # Two-way price for flat rates
                        breakdown.append(f"📌 Flat rate up to {slab['end']} km: ₹{total_price:.2f}")
                    else:
                        total_price += slab_length * slab["rate"] * 2  # Two-way price for per km rates
                        breakdown.append(f"📌 {slab_length} km × ₹{slab['rate'] * 2}/km = ₹{slab_length * slab['rate'] * 2:.2f}")
                    remaining_distance -= slab_length

            # Calculate extra distance if distance exceeds last slab (using two-way price)
            if distance > slab_ends[-1]:
                extra_distance = distance - slab_ends[-1]
                total_price += extra_distance * slab_rates[-1] * 2  # Two-way price for extra distance
                breakdown.append(f"📌 Extra {extra_distance} km × ₹{slab_rates[-1] * 2}/km = ₹{extra_distance * slab_rates[-1] * 2:.2f}")

            st.info(f"🛣️ Distance: {distance} km")
            st.success(f"💸 Estimated Two-Way Price: ₹{total_price:.2f}")
            with st.expander("🔍 Price Breakdown"):
                for line in breakdown:
                    st.write(line)

            # ➕ Price Comparison Table
            if f"slab_data_{key_prefix}" in st.session_state:
                slab_starts, slab_ends, slab_rates = st.session_state[f"slab_data_{key_prefix}"]

                def compute_two_way_price(row):
                    return 2 * calculate_price_custom(row['Distance From Crusher'], slab_rates, slab_ends[:-1])

                filtered_df['Estimated_Two_Way_Price'] = filtered_df.apply(compute_two_way_price, axis=1)
                filtered_df['Difference'] = filtered_df['logistics_value'] - filtered_df['Estimated_Two_Way_Price']

                display_df = filtered_df[['Distance From Crusher', 'logistics_value', 'Estimated_Two_Way_Price', 'Difference']]
                st.subheader("🧾 Logistics Comparison Table")
                st.dataframe(display_df.style.format({
                    'logistics_value': '₹{:.2f}',
                    'Estimated_Two_Way_Price': '₹{:.2f}',
                    'Difference': '₹{:.2f}'
                }))

                # Scatter Plot for Logistic Value and Estimated Price
                fig = go.Figure()

                # Plot logistics_value (blue points)
                fig.add_trace(go.Scatter(
                    x=filtered_df['Distance From Crusher'],
                    y=filtered_df['logistics_value'],
                    mode='markers',
                    name='Logistics Value',
                    marker=dict(color='blue', size=10)
                ))

                # Plot Estimated Two-Way Price (cross-mark points)
                fig.add_trace(go.Scatter(
                    x=filtered_df['Distance From Crusher'],
                    y=filtered_df['Estimated_Two_Way_Price'],
                    mode='markers',
                    name='Estimated Price',
                    marker=dict(color='red', symbol='x', size=10)
                ))

                fig.update_layout(
                    title="Logistics Value vs Estimated Price",
                    xaxis_title="Distance (km)",
                    yaxis_title="Price (₹)",
                    template="plotly_white"
                )

                st.plotly_chart(fig)

    else:
        st.info("📥 Please upload a CSV file to begin.")


# ---------------------------- Run App

if __name__ == "__main__":
    app()
