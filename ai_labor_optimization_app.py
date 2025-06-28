
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json
from scipy.optimize import linprog

# --- Konfigurasi Streamlit ---
st.set_page_config(page_title="📊 Optimasi Tenaga Kerja & Analisis AI", layout="wide")
st.title("📊 Dashboard Analisis Biaya Tenaga Kerja & Optimasi")

# --- Fungsi Analisis AI ---
def ai_analysis(prompt):
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    API_KEY = st.secrets["openrouter_api_key"]

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "You are a helpful data analyst and Answer All in Indonesian"},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(API_URL, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"⚠️ Gagal mengambil insight AI (kode: {response.status_code})"

# --- Upload File Excel ---
uploaded_file = st.file_uploader("📤 Upload file Excel simulasi tenaga kerja", type=["xlsx"])
if uploaded_file:
    data = pd.read_excel(uploaded_file)
    data["date"] = pd.to_datetime(data["date"])

    max_hours = 8
    overtime_hours = 4
    productivity = 10

    data["regular_capacity"] = data["actual_attendance"] * max_hours * productivity
    data["workload_gap"] = data["incoming_orders"] - data["regular_capacity"]
    data["percentage_fulfilled"] = (data["regular_capacity"] / data["incoming_orders"]).clip(upper=1)

    st.subheader("📈 Beban Kerja vs Kapasitas Reguler Harian")
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(data["date"], data["incoming_orders"], label="Incoming Orders")
    ax1.plot(data["date"], data["regular_capacity"], label="Regular Capacity")
    ax1.set_title("Beban Kerja vs Kapasitas Reguler")
    ax1.set_xlabel("Tanggal")
    ax1.set_ylabel("Jumlah Order")
    ax1.legend()
    st.pyplot(fig1)

    with st.expander("📋 Analisis AI - Beban Kerja vs Kapasitas Reguler"):
        prompt = f"Berikan insight dari data beban kerja harian dan kapasitas reguler berikut:\n\n{data[['date','incoming_orders','regular_capacity']].head(20).to_string(index=False)}"
        st.markdown(ai_analysis(prompt))

    data["month"] = data["date"].dt.to_period("M")
    monthly = data.groupby("month")[["regular_cost", "overtime_cost", "total_labor_cost"]].sum()
    monthly["overtime_ratio"] = (monthly["overtime_cost"] / monthly["total_labor_cost"]).round(3)

    st.subheader("📊 Rasio Biaya Lembur terhadap Total Biaya per Bulan")
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(monthly.index.to_timestamp(), monthly["overtime_ratio"], marker="o")
    ax2.set_title("Rasio Biaya Lembur")
    ax2.set_xlabel("Bulan")
    ax2.set_ylabel("Rasio")
    st.pyplot(fig2)

    with st.expander("📋 Analisis AI - Rasio Biaya Lembur"):
        prompt = f"Berikan analisis dan rekomendasi dari tren rasio lembur berikut:\n\n{monthly[['overtime_ratio']].tail(12).to_string()}"
        st.markdown(ai_analysis(prompt))

    st.subheader("🚨 Hari-hari Kritis dengan Lembur > 40 Jam")
    critical = data[data["overtime_hours"] > 40]
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    ax3.scatter(critical["date"], critical["overtime_hours"], color="red", label="> 40 jam")
    ax3.set_title(f"Hari Kritis: {critical.shape[0]} Hari")
    ax3.set_xlabel("Tanggal")
    ax3.set_ylabel("Jam Lembur")
    st.pyplot(fig3)

    with st.expander("📋 Analisis AI - Hari Kritis"):
        prompt = f"Berikan interpretasi dari hari-hari dengan jam lembur di atas 40:\n\n{critical[['date','overtime_hours']].head(10).to_string(index=False)}"
        st.markdown(ai_analysis(prompt))

    jan_data = data[data["date"].dt.month == 1].copy().reset_index(drop=True)

    daily_orders = jan_data["incoming_orders"].values
    reg_cost = jan_data["regular_wage_per_hour"].values * max_hours
    ot_cost = jan_data["overtime_wage_per_hour"].values * overtime_hours
    total_cost = reg_cost + ot_cost
    A = -np.eye(len(jan_data)) * productivity * (max_hours + overtime_hours)
    b = -daily_orders
    bounds = [(0, None)] * len(jan_data)
    result = linprog(total_cost, A_ub=A, b_ub=b, bounds=bounds, method='highs')

    jan_data["optimal_workers"] = np.ceil(result.x).astype(int)
    jan_data["total_optimized_cost"] = jan_data["optimal_workers"] * total_cost

    st.subheader("👷 Optimasi Jumlah Pekerja vs Aktual (Januari)")
    fig4, ax4 = plt.subplots(figsize=(14, 5))
    ax4.plot(jan_data["date"], jan_data["actual_attendance"], label="Aktual")
    ax4.plot(jan_data["date"], jan_data["optimal_workers"], label="Optimal", linestyle="--")
    ax4.set_title("Perbandingan Jumlah Pekerja")
    ax4.set_ylabel("Jumlah")
    st.pyplot(fig4)

    st.subheader("💰 Biaya Aktual vs Biaya Optimal (Januari)")
    fig5, ax5 = plt.subplots(figsize=(14, 5))
    ax5.plot(jan_data["date"], jan_data["total_labor_cost"], label="Aktual")
    ax5.plot(jan_data["date"], jan_data["total_optimized_cost"], label="Optimasi", linestyle="--")
    ax5.set_ylabel("Biaya (Rp)")
    st.pyplot(fig5)

    st.subheader("📉 Penghematan Biaya Harian")
    jan_data["cost_saving"] = jan_data["total_labor_cost"] - jan_data["total_optimized_cost"]
    fig6, ax6 = plt.subplots(figsize=(14, 5))
    ax6.bar(jan_data["date"].dt.strftime('%Y-%m-%d'), jan_data["cost_saving"], color="green")
    ax6.set_ylabel("Penghematan (Rp)")
    plt.xticks(rotation=45)
    st.pyplot(fig6)

    with st.expander("📋 Analisis AI - Efisiensi Biaya"):
        prompt = f"Berikan insight dari data efisiensi biaya berikut:\n\n{jan_data[['date','total_labor_cost','total_optimized_cost','cost_saving']].head(10).to_string(index=False)}"
        st.markdown(ai_analysis(prompt))
