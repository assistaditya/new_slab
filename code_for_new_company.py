import pandas as pd
import numpy as np
from scipy.optimize import minimize
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error

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
    price += (distance - slab_distances[-1]) * slab_rates[-1]
    return price

# ---------------------------- Optimization
def optimize_variable_slab_rates(data, slab_distances, fixed_flat_rate):
    distances = data['Distance From Crusher'].values
    actual_prices = data['One_Way_Price'].values
    n_slabs = len(slab_distances)

    def objective(rates):
        slab_rates = [fixed_flat_rate] + list(rates)
        return np.sum([
            (calculate_price_custom(d, slab_rates, slab_distances) - p) ** 2
            for d, p in zip(distances, actual_prices)
        ])

    # Initial Guess
    initial_guess = [20.0, 15.0, 10.0, 5.0][:n_slabs]

    # Bounds for each rate (must be positive)
    bounds = [(1e-2, 1e4)] * n_slabs

    # Constraints for descending order
    constraints = []
    for i in range(n_slabs - 1):
        constraints.append({'type': 'ineq', 'fun': lambda rates, i=i: rates[i] - rates[i+1] - 1e-2})

    result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        return [fixed_flat_rate] + list(result.x)
    return None

# ---------------------------- RMSE
def calculate_rmse(actual_prices, predicted_prices):
    return np.sqrt(mean_squared_error(actual_prices, predicted_prices))

# ---------------------------- Result Table
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
    st.set_page_config(page_title="Reverse Slab Rate Estimation", layout="centered")
    st.title("🔁 Reverse Slab Rate Estimation")

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
        filtered_df = df[(df['company_name'] == company) & (df['quantity_value'] == quantity)].copy()

        slab_count = st.selectbox("🔢 Select Slab Count", options=[3, 4, 5], index=0)
        st.subheader("📐 Enter Slab Start & End (in KM) and Slab 1 Fixed Price")

        slab_starts = []
        slab_ends = []
        previous_end = 0

        for i in range(slab_count):
            col1, col2 = st.columns(2)
            with col1:
                start = st.number_input(f"Slab {i+1} Start (KM)", key=f"start_{i}", min_value=0, value=previous_end)
            with col2:
                end = st.number_input(
                    f"Slab {i+1} End (KM)", key=f"end_{i}",
                    min_value=start + 1, value=max(start + 10, start + 1)
                )
            slab_starts.append(start)
            slab_ends.append(end)
            previous_end = end

        # Slab 1 fixed two-way price
        price_two_way = st.number_input("🧾 Slab 1 Two-Way Price (₹)", min_value=1.0, value=500.0)
        price_one_way = price_two_way / 2
        slab_distances = slab_ends

        if st.button("🚀 Compute Remaining Slab Rates"):
            with st.spinner("⏳ Optimizing per-km slab rates..."):
                slab_rates = optimize_variable_slab_rates(filtered_df, slab_distances, price_one_way)

                if slab_rates is not None:
                    result_df = generate_result_df(company, quantity, slab_starts, slab_ends, slab_rates)
                    st.session_state['slab_data'] = (slab_starts, slab_ends, slab_rates)
                    st.dataframe(result_df)

                    # CSV Download
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 Download CSV", data=csv, file_name="reverse_slab_rates.csv", mime="text/csv")

                    # Excel Download
                    with pd.ExcelWriter("reverse_slab_rates.xlsx", engine="openpyxl") as writer:
                        result_df.to_excel(writer, index=False)
                    with open("reverse_slab_rates.xlsx", "rb") as f:
                        st.download_button(
                            "📥 Download Excel", data=f, file_name="reverse_slab_rates.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                    predicted_prices = [
                        calculate_price_custom(d, slab_rates, slab_distances) for d in filtered_df['Distance From Crusher']
                    ]
                    rmse = calculate_rmse(filtered_df['One_Way_Price'], predicted_prices)
                    st.success(f"📊 RMSE: ₹{rmse:.2f}")
                else:
                    st.error("❌ Optimization failed. Try different slab boundaries or initial values.")
    else:
        st.info("📥 Please upload a CSV file to get started.")

# ---------------------------- Run App
if __name__ == "__main__":
    app()
