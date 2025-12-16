import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ========================================
# KONFIGURASI PAGE
# ========================================
st.set_page_config(
    page_title="Prediksi Pengangguran 2023",
    page_icon="📊",
    layout="wide"
)

# ========================================
# HEADER
# ========================================
st.title("📊 Analisis dan Prediksi Pengangguran Berdasarkan Pendidikan")
st.markdown("**Sistem Prediksi Berbasis Machine Learning - Support Vector Machine (SVM)**")
st.markdown("---")

# ========================================
# PARAMETER MODEL (Default - bisa dijelaskan saat presentasi)
# ========================================
kernel_type = "rbf"  # Kernel yang digunakan
c_value = 1.0  # Nilai regularization



# ========================================
# UPLOAD FILE
# ========================================
st.header("📁 Step 1: Upload Data")
st.markdown("**Fungsi:** Memasukkan data historis pengangguran untuk dianalisis")

col1, col2 = st.columns([3, 1])

with col1:
    file = st.file_uploader(
        "Upload file CSV (format: Periode;SD;SLTP;SLTA Umum/SMU;Universitas)",
        type=["csv"]
    )

with col2:
    if file:
        st.success("✅ File OK")
        st.caption(f"📄 {file.name}")
        st.caption(f"📦 {file.size/1024:.1f} KB")

# ========================================
# CONTOH FORMAT (Jika belum upload)
# ========================================
if file is None:
    st.warning("⚠️ Belum ada file yang diupload")
    
    st.subheader("📄 Contoh Format CSV:")
    sample = pd.DataFrame({
        'Periode': [2018, 2019, 2020, 2021, 2022],
        'SD': [1208640, 1306493, 1259740, 1280000, 1275000],
        'SLTP': [1436435, 1559768, 1487278, 1500000, 1495000],
        'SLTA Umum/SMU': [2205639, 2388976, 2402634, 2420000, 2435000],
        'Universitas': [903057, 924100, 743913, 760000, 770000]
    })
    st.dataframe(sample, use_container_width=True)
    
    # Download template
    csv = sample.to_csv(index=False, sep=';').encode('utf-8')
    st.download_button("📥 Download Template", csv, "template.csv", "text/csv")
    
    st.stop()

# ========================================
# PROCESSING DATA
# ========================================
try:
    df = pd.read_csv(file, sep=';')
    df['Periode'] = df['Periode'].astype(int)
    df = df[df['Periode'].isin([2018, 2019, 2020, 2021, 2022])]
    
    df_year = df.groupby('Periode')[
        ['SD', 'SLTP', 'SLTA Umum/SMU', 'Universitas']
    ].mean().reset_index()
    
    st.markdown("---")
    
    # ========================================
    # TAMPILKAN DATA
    # ========================================
    st.header("📊 Step 2: Data Overview")
    st.markdown("**Fungsi:** Melihat data yang akan digunakan untuk training model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Tabel Data Tahunan")
        st.dataframe(
            df_year.style.format({
                'SD': '{:,.0f}',
                'SLTP': '{:,.0f}',
                'SLTA Umum/SMU': '{:,.0f}',
                'Universitas': '{:,.0f}'
            }),
            use_container_width=True
        )
    
    with col2:
        st.subheader("Statistik Ringkas")
        for col in ['SD', 'SLTP', 'SLTA Umum/SMU', 'Universitas']:
            avg_val = df_year[col].mean()
            st.metric(col, f"{avg_val:,.0f}", "rata-rata")
    
    st.markdown("---")
    
    # ========================================
    # TRAINING MODEL
    # ========================================
    st.header("🤖 Step 3: Machine Learning Training")
    st.markdown("**Fungsi:** Melatih model SVM untuk belajar pola dari data")
    
    with st.spinner("⏳ Training model..."):
        X = df_year['Periode'].values.reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        features = ['SD', 'SLTP', 'SLTA Umum/SMU', 'Universitas']
        prediksi_2023 = {}
        models = {}
        scores = {}
        
        for col in features:
            y = df_year[col].values
            model = SVR(kernel=kernel_type, C=c_value)
            model.fit(X_scaled, y)
            models[col] = model
            
            X_2023 = scaler.transform([[2023]])
            prediksi_2023[col] = model.predict(X_2023)[0]
            
            y_pred = model.predict(X_scaled)
            scores[col] = {
                'R²': r2_score(y, y_pred),
                'MAE': mean_absolute_error(y, y_pred)
            }
    
    st.success("✅ Training selesai!")
    
    # Info box untuk presentasi
    st.info("📌 **Penjelasan untuk Presentasi:** Model SVM telah mempelajari pola pengangguran dari 3 tahun terakhir (2018-2022) dan menghasilkan fungsi prediksi untuk tahun 2023.")
    
    st.markdown("---")
    
    # ========================================
    # HASIL PREDIKSI
    # ========================================
    st.header("🎯 Step 4: Hasil Prediksi Tahun 2023")
    st.markdown("**Fungsi:** Menampilkan hasil prediksi jumlah pengangguran tahun 2023")
    
    # Metrics Cards
    cols = st.columns(4)
    for idx, (pendidikan, nilai) in enumerate(prediksi_2023.items()):
        with cols[idx]:
            nilai_2022 = df_year[df_year['Periode'] == 2022][pendidikan].values[0]
            delta = nilai - nilai_2022
            delta_pct = (delta / nilai_2022) * 100
            
            cols[idx].metric(
                label=f"**{pendidikan}**",
                value=f"{nilai:,.0f}",
                delta=f"{delta_pct:+.1f}%",
                delta_color="inverse"
            )
    
    # Tabel Detail
    st.subheader("Tabel Prediksi Detail")
    hasil_df = pd.DataFrame({
        'Tingkat Pendidikan': prediksi_2023.keys(),
        'Prediksi 2023': [f"{v:,.0f}" for v in prediksi_2023.values()],
        'Data 2022': [f"{df_year[df_year['Periode']==2022][k].values[0]:,.0f}" for k in prediksi_2023.keys()],
        'Perubahan': [f"{(prediksi_2023[k] - df_year[df_year['Periode']==2022][k].values[0]):+,.0f}" for k in prediksi_2023.keys()]
    })
    st.dataframe(hasil_df, use_container_width=True)
    
    # Download button
    csv_result = hasil_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Hasil Prediksi", csv_result, "hasil_prediksi_2023.csv", "text/csv")
    
    st.info("📌 **Penjelasan untuk Presentasi:** Angka delta (%) menunjukkan perubahan dibanding tahun 2022. Warna merah = naik, hijau = turun.")
    
    st.markdown("---")
    
    # ========================================
    # VISUALISASI
    # ========================================
    st.header("📈 Step 5: Visualisasi Data")
    st.markdown("**Fungsi:** Memvisualisasikan tren dan hasil prediksi secara grafis")
    
    # VISUAL 1: Tren Historis
    st.subheader("1️⃣ Tren Pengangguran 2018-2022")
    st.caption("📌 **Kegunaan:** Melihat pola naik/turun pengangguran per tingkat pendidikan")
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for idx, col in enumerate(features):
        ax1.plot(df_year['Periode'], df_year[col], 
                marker='o', linewidth=3, markersize=10, 
                label=col, color=colors[idx])
    
    ax1.set_xlabel("Tahun", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Jumlah Pengangguran", fontsize=12, fontweight='bold')
    ax1.set_title("Tren Pengangguran Berdasarkan Pendidikan", fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(df_year['Periode'])
    plt.tight_layout()
    st.pyplot(fig1)
    
    with st.expander("💡 Cara Membaca Grafik"):
        st.write("- **Garis naik** = pengangguran meningkat")
        st.write("- **Garis turun** = pengangguran menurun")
        st.write("- **Garis datar** = pengangguran stabil")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # VISUAL 2: Prediksi 2023
    st.subheader("2️⃣ Prediksi Pengangguran 2023")
    st.caption("📌 **Kegunaan:** Membandingkan prediksi antar tingkat pendidikan")
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bars = ax2.bar(features, list(prediksi_2023.values()), 
                   color=colors, edgecolor='black', linewidth=2, alpha=0.8)
    
    # Label di atas bar
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_xlabel("Tingkat Pendidikan", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Prediksi Jumlah Pengangguran", fontsize=12, fontweight='bold')
    ax2.set_title("Perbandingan Prediksi 2023", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig2)
    
    max_pred = max(prediksi_2023.items(), key=lambda x: x[1])
    min_pred = min(prediksi_2023.items(), key=lambda x: x[1])
    st.info(f"📌 **Insight:** Tertinggi = {max_pred[0]} ({max_pred[1]:,.0f}), Terendah = {min_pred[0]} ({min_pred[1]:,.0f})")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # VISUAL 3: Tren + Prediksi
    st.subheader("3️⃣ Tren Historis + Prediksi 2023")
    st.caption("📌 **Kegunaan:** Melihat apakah prediksi masuk akal dengan tren yang ada")
    
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    
    for idx, col in enumerate(features):
        # Data aktual
        ax3.plot(df_year['Periode'], df_year[col],
                marker='o', linewidth=3, markersize=10,
                label=f"{col}", color=colors[idx])
        
        # Prediksi 2023
        ax3.scatter(2023, prediksi_2023[col],
                   marker='*', s=400, color=colors[idx],
                   edgecolors='black', linewidths=2, zorder=5)
    
    ax3.axvline(x=2022.5, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax3.text(2022.5, ax3.get_ylim()[1]*0.95, 'Prediksi →', 
            ha='center', fontsize=10, color='red', fontweight='bold')
    
    ax3.set_xlabel("Tahun", fontsize=12, fontweight='bold')
    ax3.set_ylabel("Jumlah Pengangguran", fontsize=12, fontweight='bold')
    ax3.set_title("Tren & Prediksi (★ = Prediksi 2023)", fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks([2018, 2019, 2020, 2021, 2022, 2023])
    plt.tight_layout()
    st.pyplot(fig3)
    
    st.info("📌 **Penjelasan:** Bintang (★) menunjukkan titik prediksi 2023. Garis merah = batas antara data aktual dan prediksi.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # VISUAL 4: Model SVM Detail
    st.subheader("4️⃣ Visualisasi Model SVM")
    st.caption("📌 **Kegunaan:** Menunjukkan bagaimana model SVM belajar dan membuat prediksi")
    
    fig4, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(features):
        y = df_year[col].values
        model = models[col]
        
        # Garis prediksi smooth
        X_plot = np.linspace(2020, 2023, 100).reshape(-1, 1)
        X_plot_scaled = scaler.transform(X_plot)
        y_plot = model.predict(X_plot_scaled)
        
        # Plot
        axes[idx].scatter(df_year['Periode'], y, 
                         color='blue', s=150, label='Data Aktual', 
                         zorder=3, edgecolors='black', linewidths=2)
        
        axes[idx].plot(X_plot, y_plot, 
                      color='red', linewidth=3, label='Model SVM', zorder=2)
        
        axes[idx].scatter(2023, prediksi_2023[col], 
                         color='green', s=300, marker='*', 
                         label='Prediksi 2023', zorder=4,
                         edgecolors='black', linewidths=2)
        
        axes[idx].set_xlabel('Tahun', fontweight='bold')
        axes[idx].set_ylabel('Jumlah Pengangguran', fontweight='bold')
        axes[idx].set_title(f'{col} (R² = {scores[col]["R²"]:.3f})', 
                          fontweight='bold', fontsize=11)
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xticks([2018, 2019, 2020, 2021, 2022, 2023])
    
    plt.tight_layout()
    st.pyplot(fig4)
    
    st.info("""
    📌 **Penjelasan Elemen:**
    - **Titik Biru (●)** = Data aktual untuk training
    - **Garis Merah (—)** = Fungsi yang dipelajari model SVM
    - **Bintang Hijau (★)** = Hasil prediksi 2023
    - **R² Score** = Akurasi model (mendekati 1.0 = sangat baik)
    """)
    
    st.markdown("---")
    
    # ========================================
    # EVALUASI MODEL
    # ========================================
    st.header("📊 Step 6: Evaluasi Performa Model")
    st.markdown("**Fungsi:** Mengukur seberapa baik model bekerja")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Metrik Evaluasi")
        eval_data = []
        for col in features:
            eval_data.append({
                'Pendidikan': col,
                'R² Score': f"{scores[col]['R²']:.4f}",
                'MAE': f"{scores[col]['MAE']:.2f}"
            })
        
        eval_df = pd.DataFrame(eval_data)
        st.dataframe(eval_df, use_container_width=True)
        
        with st.expander("📚 Penjelasan Metrik"):
            st.markdown("""
            **R² Score (0-1):**
            - > 0.9 = Excellent (model sangat baik)
            - 0.7-0.9 = Good (model baik)
            - 0.5-0.7 = Fair (model cukup)
            - < 0.5 = Poor (model kurang baik)
            
            **MAE (Mean Absolute Error):**
            - Rata-rata selisih antara prediksi dan aktual
            - Semakin kecil semakin baik
            """)
    
    with col2:
        st.subheader("Parameter Model")
        st.markdown(f"""
        - **Kernel:** {kernel_type.upper()}
        - **C Value:** {c_value}
        - **Training Data:** 2018-2022
        - **Target:** 2023
        """)
        
        avg_r2 = np.mean([scores[col]['R²'] for col in features])
        st.metric("Rata-rata R²", f"{avg_r2:.4f}")
        
        if avg_r2 > 0.9:
            st.success("✅ Model Excellent!")
        elif avg_r2 > 0.7:
            st.success("✅ Model Good!")
        else:
            st.warning("⚠️ Model perlu improvement")
    
    st.markdown("---")
    
    # ========================================
    # KESIMPULAN
    # ========================================
    st.header("💡 Kesimpulan & Rekomendasi")
    st.markdown("**Fungsi:** Ringkasan hasil analisis dan saran kebijakan")
    
    col1, col2 = st.columns(2)
    
    best_r2 = max(scores.items(), key=lambda x: x[1]['R²'])
    worst_r2 = min(scores.items(), key=lambda x: x[1]['R²'])
    
    with col1:
        st.success(f"""
        **✅ Model Terbaik:**
        
        **{best_r2[0]}**
        - R² Score: {best_r2[1]['R²']:.4f}
        - Paling akurat
        """)
    
    with col2:
        st.warning(f"""
        **⚠️ Perlu Perhatian:**
        
        **{max_pred[0]}**
        - Prediksi tertinggi: {max_pred[1]:,.0f}
        - Prioritas kebijakan
        """)
    
    st.subheader("📋 Rekomendasi Kebijakan:")
    st.markdown(f"""
    1. **{max_pred[0]}:** Fokus program pelatihan kerja karena prediksi tertinggi
    2. **{worst_r2[0]}:** Tambah data historis untuk akurasi lebih baik
    3. **Monitoring:** Evaluasi prediksi vs aktual setiap tahun
    4. **Intervensi:** Program khusus untuk tingkat pendidikan dengan tren naik
    """)
    
    st.markdown("---")
    
    

except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.info("Pastikan format file CSV benar")