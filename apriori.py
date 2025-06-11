import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('D:/TB PAVD & DATA MINING/dataset/Transaksii_Petshop.csv')
df['Tanggal Transaksi'] = pd.to_datetime(df['Tanggal Transaksi'], format="%Y-%m-%d")

# Judul aplikasi
st.markdown("#### 🚀 Selamat Datang di Aplikasi Prediksi Pembelian Petshop Menggunakan Algoritma Apriori!")
st.divider()

# Menampilkan statistik dataset
st.markdown("### Statistik Dataset")
if df.empty:
    st.error("Dataset kosong. Pastikan file dataset sudah benar.")
else:
    st.write(f"Jumlah total transaksi: {df['ID Transaksi'].nunique()}")
    st.write(f"Rentang tanggal transaksi: {df['Tanggal Transaksi'].min()} hingga {df['Tanggal Transaksi'].max()}")

# Pilihan item yang dibeli user
item_columns = [col for col in df.columns if "Nama Barang" in col]
st.markdown("### Pilih Barang yang Dibeli")
item_dipilih = {}
for item_col in item_columns:
    item_dipilih[item_col] = st.selectbox(
        f"Barang pada {item_col}",
        ["Tidak ada"] + df[item_col].dropna().unique().tolist(),
        key=item_col
    )

# Membentuk filter data berdasarkan item yang dipilih
def filter_by_items(data, items):
    for col, val in items.items():
        if val != "Tidak ada":
            data = data[data[col] == val]
    return data

data = filter_by_items(df, item_dipilih)

# Fungsi encoding
def encode(x):
    return 1 if x >= 1 else 0

def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Proses analisis aturan asosiasi
if not data.empty:
    # Membentuk data pivot
    data_melted = data.melt(id_vars=["ID Transaksi"], value_vars=item_columns, var_name="Barang Ke", value_name="Item")
    data_melted.dropna(subset=["Item"], inplace=True)
    item_count = data_melted.groupby(["ID Transaksi", "Item"]).size().reset_index(name="Count")
    item_count_pivot = item_count.pivot_table(index="ID Transaksi", columns="Item", values="Count", aggfunc='sum').fillna(0)
    item_count_pivot = item_count_pivot.applymap(encode)

    # Apriori dan aturan asosiasi
    frequent_items = apriori(item_count_pivot, min_support=0.01, use_colnames=True)
    # rules = association_rules(frequent_items, metric="lift", min_threshold=1)
    rules = association_rules(frequent_items, metric="lift", min_threshold=1, num_itemsets=len(frequent_items))
    rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]]

    # Fungsi parsing
    def parse_list(x):
        x = list(x)
        return ", ".join(x) if len(x) > 1 else x[0]

    rules["antecedents"] = rules["antecedents"].apply(parse_list)
    rules["consequents"] = rules["consequents"].apply(parse_list)

    # Filter aturan berdasarkan barang yang dipilih user
    selected_items = [val for val in item_dipilih.values() if val != "Tidak ada"]
    if selected_items:
        filtered_rules = rules[rules["antecedents"].str.contains('|'.join(selected_items))]

        st.markdown("### Hasil Rekomendasi Pembelian")
        if not filtered_rules.empty:
            st.table(filtered_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
        else:
            st.warning("Tidak ada rekomendasi yang ditemukan berdasarkan barang yang Anda pilih.")

        # Penjelasan tentang output
        st.markdown("""
        ### Penjelasan tentang Output:
        - **Antecedents**: Barang yang dibeli terlebih dahulu yang menjadi dasar rekomendasi.
        - **Consequents**: Barang yang kemungkinan besar akan dibeli setelah antecedent.
        - **Support**: Menunjukkan seberapa sering kombinasi antara antecedents dan consequents muncul dalam dataset. Semakin tinggi nilai support, semakin banyak transaksi yang melibatkan kedua item tersebut.
        - **Confidence**: Mengukur seberapa besar kemungkinan suatu barang (consequents) akan dibeli setelah membeli barang lain (antecedents). Semakin tinggi confidence, semakin kuat asosiasi antara kedua barang tersebut.
        - **Lift**: Mengukur seberapa kuat asosiasi antara antecedents dan consequents dibandingkan dengan jika kedua barang dibeli secara independen. Nilai lift lebih dari 1 menunjukkan asosiasi positif antara kedua barang.

        Semakin tinggi nilai **Lift**, **Confidence**, dan **Support**, semakin kuat hubungan antar barang dalam transaksi dan semakin baik rekomendasi yang diberikan.
        """)
        
        # Statistik aturan asosiasi (dipindahkan ke bawah hasil rekomendasi)
        st.markdown(f"### Statistik Aturan Asosiasi")
        st.write(f"Rata-rata Confidence: {rules['confidence'].mean():.2f}")
        st.write(f"Rata-rata Lift: {rules['lift'].mean():.2f}")
        st.write(f"Rata-rata Support: {rules['support'].mean():.2f}")
        st.write(f"Jumlah total aturan asosiasi: {len(rules)}")

        # Visualisasi aturan asosiasi
        if not filtered_rules.empty:
            st.markdown("### Visualisasi Aturan Asosiasi")
            fig, ax = plt.subplots()
            sns.scatterplot(data=filtered_rules, x="support", y="confidence", size="lift", hue="lift", palette="viridis", ax=ax)
            ax.set_title("Visualisasi Aturan Asosiasi")
            st.pyplot(fig)

        # Tombol untuk download hasil rekomendasi
        csv = to_csv(filtered_rules)
        st.download_button(
            label="Download Hasil Rekomendasi",
            data=csv,
            file_name="rekomendasi.csv",
            mime="text/csv"
        )
    else:
        st.warning("Silakan pilih barang terlebih dahulu untuk mendapatkan rekomendasi.")
else:
    st.warning("Tidak ada data transaksi yang sesuai dengan filter Anda.")
