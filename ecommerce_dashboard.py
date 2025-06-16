import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px # Import Plotly Express

# --- Konfigurasi Halaman Streamlit ---
st.set_page_config(
    page_title="Super Dashboard Analisis E-commerce",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fungsi untuk Memuat dan Memproses Data (dengan caching) ---
@st.cache_data
def load_and_process_data(file_path):
    try:
        ecommerce_df = pd.read_csv(file_path, low_memory=False)
    except FileNotFoundError:
        st.error(f"❌ **ERROR:** File tidak ditemukan di jalur: `{file_path}`")
        st.info("Pastikan Anda telah mengunggah file `Pakistan Largest Ecommerce Dataset.csv` di lokasi yang benar atau perbarui `FILE_PATH`.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat data: {e}")
        st.stop()

    # Membuat subset tabel-tabel berdasarkan kolom yang relevan
    customers_df = ecommerce_df[['Customer ID', 'sales_commission_code', 'Customer Since']].drop_duplicates().copy()
    orders_df = ecommerce_df[['increment_id', 'created_at', 'status', 'payment_method', 'grand_total', 'Customer ID']].drop_duplicates().copy()
    products_df = ecommerce_df[['item_id', 'sku', 'category_name_1', 'price']].drop_duplicates().copy()
    sales_df = ecommerce_df[['increment_id', 'item_id', 'qty_ordered', 'price', 'discount_amount']].drop_duplicates().copy()

    # --- Pembersihan Data Esensial ---
    orders_df['created_at'] = pd.to_datetime(orders_df['created_at'], errors='coerce')
    orders_df['grand_total'] = pd.to_numeric(orders_df['grand_total'], errors='coerce').fillna(0)
    orders_df['Customer ID'] = pd.to_numeric(orders_df['Customer ID'], errors='coerce').fillna(-1).astype(int)

    products_df.dropna(subset=['item_id', 'sku', 'category_name_1', 'price'], inplace=True)
    products_df['item_id'] = products_df['item_id'].astype(int)
    products_df['price'] = pd.to_numeric(products_df['price'], errors='coerce').fillna(0)
    products_df['sku'] = products_df['sku'].astype(str).str.upper().str.strip()
    products_df['category_name_1'] = products_df['category_name_1'].astype(str).str.upper().str.strip().replace(r'\N', 'UNKNOWN')
    products_df = products_df[products_df['price'] >= 0]

    sales_df.dropna(subset=['increment_id', 'item_id', 'qty_ordered', 'price', 'discount_amount'], inplace=True)
    sales_df['item_id'] = sales_df['item_id'].astype(int)
    sales_df['qty_ordered'] = sales_df['qty_ordered'].astype(int)
    sales_df['price'] = pd.to_numeric(sales_df['price'], errors='coerce').fillna(0)
    sales_df['discount_amount'] = pd.to_numeric(sales_df['discount_amount'], errors='coerce').fillna(0)
    sales_df = sales_df[sales_df['qty_ordered'] > 0]
    sales_df = sales_df[sales_df['price'] >= 0]
    sales_df = sales_df[sales_df['discount_amount'] >= 0]

    customers_df['Customer Since'] = pd.to_datetime(customers_df['Customer Since'], errors='coerce')
    customers_df['Customer ID'] = pd.to_numeric(customers_df['Customer ID'], errors='coerce').fillna(-1).astype(int)
    customers_df.dropna(subset=['Customer ID'], inplace=True)

    # --- Penggabungan Data ---
    customers_orders_df = pd.merge(orders_df, customers_df, how="outer", on="Customer ID")
    customer_order_sales_df = pd.merge(customers_orders_df, sales_df, how="inner", on="increment_id", suffixes=('_order', '_sale_item'))
    df_merged = pd.merge(customer_order_sales_df, products_df, how="inner", on="item_id", suffixes=('_sales_detail', '_product_info'))

    # Ganti nama kolom harga
    df_merged.rename(columns={
        'price_sales_detail': 'price_per_unit_sold',
        'price_product_info': 'product_original_price',
    }, inplace=True)

    # Feature Engineering
    df_merged['total_price_per_item'] = df_merged['qty_ordered'] * df_merged['price_per_unit_sold']
    df_merged['net_item_sales'] = df_merged['total_price_per_item'] - df_merged['discount_amount']

    # Menambahkan kolom waktu untuk EDA
    df_merged['order_date'] = df_merged['created_at'].dt.date
    df_merged['order_month'] = df_merged['created_at'].dt.to_period('M')
    df_merged['customer_since_year'] = df_merged['created_at'].dt.year # Menggunakan created_at agar selalu ada tahunnya
    df_merged['day_of_week'] = df_merged['created_at'].dt.day_name()
    df_merged['hour_of_day'] = df_merged['created_at'].dt.hour

    return df_merged

# --- Jalur File Dataset ---
FILE_PATH = "C:/Users/ASUS/Documents/Analisis_Data/Pakistan Largest Ecommerce Dataset.csv"

# Muat dan proses data
df_merged_raw = load_and_process_data(FILE_PATH) # Menyimpan raw sebelum filter

# --- Judul Aplikasi Utama ---
st.title("📈 Dashboard Analisis Penjualan E-commerce Pakistan")
st.markdown("""
Selamat datang di *dashboard* interaktif yang memukau ini!
Jelajahi *insight* mendalam dari data penjualan e-commerce dengan visualisasi yang intuitif dan terorganisir.
""")

# --- Sidebar untuk Filter ---
st.sidebar.header("⚙️ Filter Data & Opsi")

# Filter Tanggal
min_date = df_merged_raw['order_date'].min()
max_date = df_merged_raw['order_date'].max()

date_range = st.sidebar.date_input(
    "📅 Pilih Rentang Tanggal Pesanan:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    format="DD/MM/YYYY"
)

# Pastikan date_range memiliki 2 elemen
if len(date_range) == 2:
    start_date, end_date = date_range
    df_filtered = df_merged_raw[(df_merged_raw['order_date'] >= start_date) & (df_merged_raw['order_date'] <= end_date)]
else:
    df_filtered = df_merged_raw.copy() # Jika hanya 1 tanggal dipilih atau tidak valid

# Filter Kategori Produk
all_categories = df_filtered['category_name_1'].unique().tolist()
selected_categories = st.sidebar.multiselect(
    "🛒 Filter berdasarkan Kategori Produk:",
    options=all_categories,
    default=all_categories # Defaultnya semua kategori terpilih
)

if selected_categories:
    df_filtered = df_filtered[df_filtered['category_name_1'].isin(selected_categories)]
else:
    st.sidebar.warning("Silakan pilih setidaknya satu kategori produk.")
    df_filtered = pd.DataFrame() # Kosongkan DataFrame jika tidak ada kategori yang dipilih

# Tampilkan info filter di sidebar
if not df_filtered.empty:
    st.sidebar.markdown(f"**Data difilter:**")
    st.sidebar.markdown(f"**{len(df_filtered):,}** item penjualan.")
    st.sidebar.markdown(f"**{df_filtered['increment_id'].nunique():,}** pesanan unik.")
    st.sidebar.markdown(f"**{df_filtered['Customer ID'].nunique():,}** pelanggan unik.")
else:
    st.sidebar.markdown("**Tidak ada data yang cocok dengan filter yang dipilih.**")

st.sidebar.write("---")



# --- Bagian Metrik Kunci (KPIs) ---
st.header("📊  Ringkasan Performa Kunci ")

if not df_filtered.empty:
    # Menghitung metrik dari data yang difilter
    total_penjualan_bersih = df_filtered['net_item_sales'].sum()
    jumlah_pesanan_unik = df_filtered['increment_id'].nunique()
    jumlah_pelanggan_unik = df_filtered['Customer ID'].nunique()
    jumlah_produk_unik = df_filtered['sku'].nunique()
    rerata_nilai_pesanan = df_filtered.groupby('increment_id')['net_item_sales'].sum().mean() if jumlah_pesanan_unik > 0 else 0
    total_diskon_diberikan = df_filtered['discount_amount'].sum()
    # Menghindari pembagian oleh nol jika tidak ada item terjual
    rerata_diskon_per_item = df_filtered['discount_amount'].mean() if len(df_filtered) > 0 else 0
    # Untuk persentase diskon, perlu total harga sebelum diskon
    total_harga_sebelum_diskon = df_filtered['total_price_per_item'].sum()
    persentase_diskon = (total_diskon_diberikan / total_harga_sebelum_diskon) * 100 if total_harga_sebelum_diskon > 0 else 0


    col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)

    with col_kpi1:
        st.metric(label="💰 Total Penjualan Bersih", value=f"Rp {total_penjualan_bersih:,.0f}")
    with col_kpi2:
        st.metric(label="📦 Jumlah Pesanan Unik", value=f"{jumlah_pesanan_unik:,}")
    with col_kpi3:
        st.metric(label="🧑‍🤝‍🧑 Jumlah Pelanggan Unik", value=f"{jumlah_pelanggan_unik:,}")
    with col_kpi4:
        st.metric(label="🛒 Rerata Nilai Pesanan", value=f"Rp {rerata_nilai_pesanan:,.0f}")
    with col_kpi5:
        st.metric(label="📉 Total Diskon Diberikan", value=f"Rp {total_diskon_diberikan:,.0f}")

else:
    st.warning("Tidak ada data untuk menampilkan metrik. Sesuaikan filter Anda.")

st.write("---") # Garis pemisah visual

# --- Struktur Dashboard dengan Tabs ---
if not df_filtered.empty:
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "💡 Gambaran Umum Data",
        "🧑‍🤝‍🧑 Analisis Pelanggan",
        "📦 Analisis Pesanan",
        "🏷️ Analisis Produk",
        "🔗 Hubungan Variabel",
        # "📄 Data Mentah",
        "❓ Wawasan Cepat & FAQ"
    ])

    with tab1:
        st.header("💡 Gambaran Umum Data")
        st.markdown("Berikut adalah beberapa baris pertama dan ringkasan statistik dari DataFrame gabungan.")

        st.subheader("Data Gabungan (`df_merged.head()`)")
        st.dataframe(df_filtered.head(10)) # Menampilkan 10 baris pertama

        with st.expander("Lihat Informasi Lengkap DataFrame (`df_merged.info()`)"):
            # info() tidak mengembalikan DataFrame, jadi kita tangkap outputnya
            import io
            buffer = io.StringIO()
            df_filtered.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

        with st.expander("Lihat Statistik Deskriptif (`df_merged.describe()`)"):
            st.dataframe(df_filtered.describe())

    with tab2:
        st.header("🧑‍🤝‍🧑 Analisis Pelanggan")
        st.markdown("Menganalisis karakteristik pelanggan dan mengidentifikasi pelanggan paling berharga.")

        st.subheader("Distribusi Tahun Customer Sejak Bergabung")
        # Mengubah ke Plotly Express
        customer_since_counts = df_filtered['customer_since_year'].value_counts().sort_index().reset_index()
        customer_since_counts.columns = ['Tahun Bergabung', 'Jumlah Pesanan']
        fig = px.bar(customer_since_counts, x='Tahun Bergabung', y='Jumlah Pesanan',
                     title='Distribusi Tahun Customer Sejak Bergabung',
                     color_discrete_sequence=px.colors.sequential.Viridis,
                     labels={'Tahun Bergabung': 'Tahun Bergabung Customer', 'Jumlah Pesanan': 'Jumlah Pesanan'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("📝 Penjelasan Grafik: Distribusi Tahun Bergabung"):
            st.markdown("""
            Grafik batang ini menunjukkan sebaran tahun di mana pelanggan pertama kali melakukan pesanan.
            Anda dapat melihat seberapa banyak aktivitas pesanan yang berasal dari pelanggan yang bergabung pada tahun tertentu.
            **Insight:** Perhatikan tahun-tahun dengan jumlah pesanan tertinggi untuk memahami periode pertumbuhan pelanggan yang signifikan.
            """)

        st.subheader("Top 10 Pelanggan (Berdasarkan Penjualan Bersih)")
        top_customers = df_filtered.groupby('Customer ID')['net_item_sales'].sum().sort_values(ascending=False).head(10)
        # Mengubah ke Plotly Express
        top_customers_df = top_customers.reset_index()
        top_customers_df.columns = ['Customer ID', 'Total Penjualan Bersih (Rp)']
        fig_top_cust = px.bar(top_customers_df, x='Total Penjualan Bersih (Rp)', y='Customer ID', orientation='h',
                              title='Top 10 Pelanggan (Berdasarkan Penjualan Bersih)',
                              color_discrete_sequence=px.colors.sequential.YlOrRd,
                              labels={'Total Penjualan Bersih (Rp)': 'Total Penjualan Bersih (Rp)'})
        fig_top_cust.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_top_cust, use_container_width=True)
        with st.expander("📝 Penjelasan Grafik: Top 10 Pelanggan"):
            st.markdown("""
            Grafik ini menampilkan 10 ID pelanggan dengan kontribusi penjualan bersih tertinggi.
            **Insight:** Identifikasi pelanggan paling berharga Anda. Anda dapat menggunakan informasi ini untuk program loyalitas atau strategi pemasaran yang ditargetkan.
            """)


    with tab3:
        st.header("📦 Analisis Pesanan")
        st.markdown("Menggali wawasan tentang status pesanan, metode pembayaran, dan tren penjualan dari waktu ke waktu.")

        col_order_dist1, col_order_dist2 = st.columns(2)
        with col_order_dist1:
            st.subheader("Distribusi Status Pesanan")
            status_counts = df_filtered['status'].value_counts().reset_index()
            status_counts.columns = ['Status Pesanan', 'Jumlah Item Pesanan']
            fig1 = px.pie(status_counts, values='Jumlah Item Pesanan', names='Status Pesanan',
                          title='Distribusi Status Pesanan', hole=0.3,
                          color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig1, use_container_width=True)
            with st.expander("📝 Penjelasan Grafik: Distribusi Status Pesanan"):
                st.markdown("""
                Diagram lingkaran ini menunjukkan proporsi item pesanan berdasarkan statusnya (misalnya, `complete`, `pending`, `canceled`).
                **Insight:** Ini membantu Anda memahami efisiensi pemrosesan pesanan dan area mana yang mungkin memerlukan perhatian (misalnya, banyak pesanan `pending` atau `canceled`).
                """)

        with col_order_dist2:
            st.subheader("Distribusi Metode Pembayaran")
            payment_counts = df_filtered['payment_method'].value_counts().reset_index()
            payment_counts.columns = ['Metode Pembayaran', 'Jumlah Item Pesanan']
            fig2 = px.bar(payment_counts, x='Metode Pembayaran', y='Jumlah Item Pesanan',
                          title='Distribusi Metode Pembayaran',
                          color_discrete_sequence=px.colors.sequential.YlGnBu)
            st.plotly_chart(fig2, use_container_width=True)
            with st.expander("📝 Penjelasan Grafik: Distribusi Metode Pembayaran"):
                st.markdown("""
                Grafik batang ini menampilkan seberapa sering setiap metode pembayaran digunakan.
                **Insight:** Pahami preferensi pelanggan dalam pembayaran, yang dapat memandu keputusan terkait opsi pembayaran dan promosi.
                """)

        st.subheader("Tren Penjualan Bersih Seiring Waktu")
        col_time_trend1, col_time_trend2 = st.columns(2)

        with col_time_trend1:
            monthly_net_sales = df_filtered.groupby('order_month')['net_item_sales'].sum().reset_index()
            # KOREKSI: Konversi kolom 'order_month' (Period) menjadi string agar dapat di-serialisasi JSON
            monthly_net_sales['order_month'] = monthly_net_sales['order_month'].astype(str)
            monthly_net_sales.columns = ['Bulan', 'Total Penjualan Bersih (Rp)']
            fig5 = px.line(monthly_net_sales, x='Bulan', y='Total Penjualan Bersih (Rp)',
                           title='Tren Penjualan Bersih Bulanan',
                           markers=True, line_shape="spline",
                           color_discrete_sequence=['teal'])
            fig5.update_xaxes(type='category') # Ensure months are treated as categories for ordering
            st.plotly_chart(fig5, use_container_width=True)
            with st.expander("📝 Penjelasan Grafik: Tren Penjualan Bulanan"):
                st.markdown("""
                Grafik garis ini menunjukkan total penjualan bersih dari waktu ke waktu setiap bulannya.
                **Insight:** Identifikasi periode puncak penjualan dan penurunan, yang penting untuk perencanaan inventaris dan kampanye pemasaran.
                """)

        with col_time_trend2:
            daily_net_sales = df_filtered.groupby(df_filtered['created_at'].dt.date)['net_item_sales'].sum().reset_index()
            daily_net_sales.columns = ['Tanggal', 'Total Penjualan Bersih (Rp)']
            fig6 = px.line(daily_net_sales, x='Tanggal', y='Total Penjualan Bersih (Rp)',
                           title='Tren Penjualan Bersih Harian',
                           line_shape="spline",
                           color_discrete_sequence=['blue'])
            st.plotly_chart(fig6, use_container_width=True)
            with st.expander("📝 Penjelasan Grafik: Tren Penjualan Harian"):
                st.markdown("""
                Grafik garis ini menunjukkan total penjualan bersih dari waktu ke waktu setiap harinya.
                **Insight:** Perhatikan pola penjualan harian, seperti hari-hari tertentu dalam seminggu yang menunjukkan peningkatan atau penurunan penjualan.
                """)

        st.subheader("Pola Penjualan Berdasarkan Hari & Jam")
        col_day_hour1, col_day_hour2 = st.columns(2)

        with col_day_hour1:
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sales_by_day = df_filtered.groupby('day_of_week')['net_item_sales'].sum().reindex(day_order).reset_index()
            sales_by_day.columns = ['Hari dalam Seminggu', 'Total Penjualan Bersih (Rp)']
            fig_day = px.bar(sales_by_day, x='Hari dalam Seminggu', y='Total Penjualan Bersih (Rp)',
                             title='Total Penjualan Bersih per Hari dalam Seminggu',
                             color_discrete_sequence=px.colors.sequential.Plasma)
            st.plotly_chart(fig_day, use_container_width=True)
            with st.expander("📝 Penjelasan Grafik: Penjualan per Hari"):
                st.markdown("""
                Grafik ini menunjukkan hari-hari dalam seminggu dengan penjualan bersih tertinggi.
                **Insight:** Gunakan informasi ini untuk menjadwalkan promosi atau alokasi sumber daya pada hari-hari puncak.
                """)

        with col_day_hour2:
            sales_by_hour = df_filtered.groupby('hour_of_day')['net_item_sales'].sum().reset_index()
            sales_by_hour.columns = ['Jam dalam Sehari', 'Total Penjualan Bersih (Rp)']
            fig_hour = px.line(sales_by_hour, x='Jam dalam Sehari', y='Total Penjualan Bersih (Rp)',
                               title='Total Penjualan Bersih per Jam dalam Sehari',
                               markers=True, line_shape="spline",
                               color_discrete_sequence=['purple'])
            st.plotly_chart(fig_hour, use_container_width=True)
            with st.expander("📝 Penjelasan Grafik: Penjualan per Jam"):
                st.markdown("""
                Grafik ini menunjukkan jam-jam dalam sehari dengan penjualan bersih tertinggi.
                **Insight:** Pahami kapan pelanggan paling aktif untuk mengoptimalkan kampanye iklan dan waktu operasional.
                """)


    with tab4:
        st.header("🏷️ Analisis Produk")
        st.markdown("Mengidentifikasi kategori produk paling populer dan produk terlaris berdasarkan penjualan.")

        col_prod_cat1, col_prod_cat2 = st.columns(2)

        with col_prod_cat1:
            st.subheader("Top 10 Kategori Produk (Jumlah Item Terjual)")
            top_cat_counts = df_filtered['category_name_1'].value_counts().head(10).reset_index()
            top_cat_counts.columns = ['Kategori Produk', 'Jumlah Item Terjual']
            fig7 = px.bar(top_cat_counts, x='Jumlah Item Terjual', y='Kategori Produk', orientation='h',
                          title='Top 10 Kategori Produk (Jumlah Item Terjual)',
                          color_discrete_sequence=px.colors.sequential.OrRd)
            fig7.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig7, use_container_width=True)
            with st.expander("📝 Penjelasan Grafik: Top Kategori (Jumlah Item)"):
                st.markdown("""
                Grafik ini menunjukkan kategori produk yang paling banyak terjual dalam hal jumlah item.
                **Insight:** Ini bisa menjadi indikator popularitas atau permintaan tinggi untuk kategori tersebut.
                """)

        with col_prod_cat2:
            st.subheader("Top 10 Kategori Produk (Total Penjualan Bersih)")
            category_net_sales_df = df_filtered.groupby('category_name_1')['net_item_sales'].sum().sort_values(ascending=False).head(10).reset_index()
            category_net_sales_df.columns = ['Kategori Produk', 'Total Penjualan Bersih (Rp)']
            fig8 = px.bar(category_net_sales_df, x='Total Penjualan Bersih (Rp)', y='Kategori Produk', orientation='h',
                          title='Top 10 Kategori Produk (Total Penjualan Bersih)',
                          color_discrete_sequence=px.colors.sequential.YlGnBu)
            fig8.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig8, use_container_width=True)
            with st.expander("📝 Penjelasan Grafik: Top Kategori (Penjualan Bersih)"):
                st.markdown("""
                Grafik ini menampilkan kategori produk yang menghasilkan pendapatan penjualan bersih tertinggi.
                **Insight:** Bandingkan dengan jumlah item terjual. Kategori yang memiliki penjualan bersih tinggi tetapi jumlah item terjual sedang mungkin memiliki harga produk yang lebih tinggi.
                """)

        st.subheader("Top 10 SKU Terlaris (Berdasarkan Total Penjualan Bersih)")
        top_sku_net_sales = df_filtered.groupby('sku')['net_item_sales'].sum().sort_values(ascending=False).head(10).reset_index()
        top_sku_net_sales.columns = ['SKU Produk', 'Total Penjualan Bersih (Rp)']
        fig9 = px.bar(top_sku_net_sales, x='Total Penjualan Bersih (Rp)', y='SKU Produk', orientation='h',
                      title='Top 10 Produk Terlaris (Berdasarkan Total Penjualan Bersih)',
                      color_discrete_sequence=px.colors.sequential.Viridis)
        fig9.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig9, use_container_width=True)
        with st.expander("📝 Penjelasan Grafik: Top 10 SKU Terlaris"):
            st.markdown("""
            Grafik ini menunjukkan produk (SKU) individual yang menghasilkan penjualan bersih tertinggi.
            **Insight:** Identifikasi produk-produk unggulan yang dapat Anda fokuskan untuk promosi atau pengembangan produk lebih lanjut.
            """)


    with tab5:
        st.header("🔗 Hubungan Antar Variabel")
        st.markdown("Mengeksplorasi potensi korelasi atau pola antara harga produk dan diskon yang diberikan.")

        st.subheader("Hubungan Harga Asli Produk vs. Jumlah Diskon")
        fig10 = px.scatter(df_filtered, x='product_original_price', y='discount_amount',
                           title='Hubungan Harga Asli Produk vs. Jumlah Diskon',
                           color='category_name_1', # Menambahkan warna berdasarkan kategori
                           hover_name='sku', # Menampilkan SKU saat hover
                           opacity=0.5,
                           labels={'product_original_price': 'Harga Asli Produk (Rp)', 'discount_amount': 'Jumlah Diskon (Rp)'},
                           range_x=[0, df_filtered['product_original_price'].quantile(0.99)], # Batasi sumbu X
                           range_y=[0, df_filtered['discount_amount'].quantile(0.99)]) # Batasi sumbu Y
        st.plotly_chart(fig10, use_container_width=True)
        with st.expander("📝 Penjelasan Grafik: Harga vs. Diskon"):
            st.markdown("""
            Grafik *scatter* ini memvisualisasikan hubungan antara harga asli produk dan jumlah diskon yang diberikan.
            **Insight:** Perhatikan apakah ada tren, misalnya, produk yang lebih mahal cenderung mendapatkan diskon lebih besar.
            Titik-titik yang berkelompok mungkin menunjukkan strategi diskon yang konsisten untuk kategori atau segmen harga tertentu.
            Anda bisa melihat detail SKU saat mengarahkan kursor (hover) pada titik.
            """)

    # with tab6:
    #     st.header("📄 Data Mentah")
    #     st.markdown("Tinjau data penjualan gabungan yang sudah difilter, bisa mencari dan mengunduh data di sini.")

    #     st.dataframe(df_filtered)

    #     # Tombol download
    #     @st.cache_data
    #     def convert_df_to_csv(df):
    #         # Cache the conversion to prevent computation on every rerun
    #         return df.to_csv(index=False).encode('utf-8')

    #     csv = convert_df_to_csv(df_filtered)

    #     st.download_button(
    #         label="Unduh Data Filtered sebagai CSV",
    #         data=csv,
    #         file_name="filtered_ecommerce_data.csv",
    #         mime="text/csv",
    #         help="Unduh data yang saat ini terlihat di tabel di atas."
    #     )

    with tab7: # TAB UNTUK WAWASAN CEPAT DAN FAQ
        st.header("❓ Wawasan Cepat & FAQ")
        st.markdown("""
        Dapatkan jawaban instan untuk pertanyaan kunci mengenai performa e-commerce Anda
        berdasarkan data yang **sedang difilter**!
        """)

        st.subheader("🚀 Insight Utama Anda Saat Ini:")

        # Pertanyaan 1: Total Penjualan Bersih
        st.markdown(f"**1. Berapa total penjualan bersih untuk data yang difilter saat ini?**")
        st.info(f"👉 Total Penjualan Bersih: **Rp {total_penjualan_bersih:,.0f}**")
        st.caption("Angka ini mencerminkan pendapatan setelah dikurangi diskon.")
        st.markdown("---")

        # Pertanyaan 2: Rata-rata Nilai Pesanan
        st.markdown(f"**2. Berapa rata-rata nilai pesanan untuk data yang difilter saat ini?**")
        st.info(f"👉 Rata-rata Nilai Pesanan: **Rp {rerata_nilai_pesanan:,.0f}**")
        st.caption("Ini adalah rata-rata pendapatan dari setiap pesanan yang berhasil.")
        st.markdown("---")

        # Pertanyaan 3: Kategori Produk Terlaris (berdasarkan penjualan bersih) - dalam bentuk grafik
        if not df_filtered.empty and 'category_name_1' in df_filtered.columns:
            st.markdown(f"**3. Kategori produk apa yang memiliki penjualan bersih tertinggi saat ini?**")
            top_category_sales_df = df_filtered.groupby('category_name_1')['net_item_sales'].sum().nlargest(3).reset_index()
            top_category_sales_df.columns = ['Kategori Produk', 'Total Penjualan Bersih (Rp)']
            fig_top_cat_faq = px.bar(top_category_sales_df, x='Total Penjualan Bersih (Rp)', y='Kategori Produk', orientation='h',
                                     title='Top 3 Kategori Terlaris',
                                     color_discrete_sequence=px.colors.sequential.YlOrBr,
                                     height=250) # Ukuran lebih kecil untuk FAQ
            fig_top_cat_faq.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=0, r=0, t=30, b=0)) # Margin agar rapi
            st.plotly_chart(fig_top_cat_faq, use_container_width=True, config={'displayModeBar': False}) # Tanpa mode bar
            st.info(f"👉 Kategori Terlaris: **{top_category_sales_df.iloc[0]['Kategori Produk']}** (Rp {top_category_sales_df.iloc[0]['Total Penjualan Bersih (Rp)']:.0f})")
            st.caption("Fokus pada kategori ini untuk potensi pertumbuhan lebih lanjut.")
            st.markdown("---")
        else:
            st.warning("Tidak dapat menentukan kategori terlaris karena data kategori tidak tersedia atau kosong.")

        # Pertanyaan 4: Metode Pembayaran Paling Populer - dalam bentuk grafik
        if not df_filtered.empty and 'payment_method' in df_filtered.columns:
            st.markdown(f"**4. Metode pembayaran apa yang paling sering digunakan oleh pelanggan?**")
            payment_counts_faq = df_filtered['payment_method'].value_counts().nlargest(3).reset_index()
            payment_counts_faq.columns = ['Metode Pembayaran', 'Jumlah Item Pesanan']
            fig_payment_faq = px.pie(payment_counts_faq, values='Jumlah Item Pesanan', names='Metode Pembayaran',
                                     title='Top 3 Metode Pembayaran', hole=0.4,
                                     color_discrete_sequence=px.colors.sequential.GnBu,
                                     height=250) # Ukuran lebih kecil untuk FAQ
            fig_payment_faq.update_layout(margin=dict(l=0, r=0, t=30, b=0)) # Margin agar rapi
            st.plotly_chart(fig_payment_faq, use_container_width=True, config={'displayModeBar': False}) # Tanpa mode bar
            st.info(f"👉 Metode Pembayaran Paling Populer: **{payment_counts_faq.iloc[0]['Metode Pembayaran']}** ({payment_counts_faq.iloc[0]['Jumlah Item Pesanan']:,} item)")
            st.caption("Pertimbangkan untuk mengoptimalkan alur untuk metode pembayaran ini.")
            st.markdown("---")
        else:
            st.warning("Tidak dapat menentukan metode pembayaran paling populer karena data metode pembayaran tidak tersedia atau kosong.")

        # Pertanyaan 5: Hari Penjualan Puncak - dalam bentuk grafik
        if not df_filtered.empty and 'day_of_week' in df_filtered.columns:
            st.markdown(f"**5. Hari apa dalam seminggu yang memiliki penjualan bersih tertinggi?**")
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            sales_by_day_faq = df_filtered.groupby('day_of_week')['net_item_sales'].sum().reindex(day_order).fillna(0).reset_index()
            sales_by_day_faq.columns = ['Hari', 'Total Penjualan Bersih (Rp)']
            fig_day_faq = px.bar(sales_by_day_faq, x='Hari', y='Total Penjualan Bersih (Rp)',
                                 title='Penjualan Bersih per Hari dalam Seminggu',
                                 color_discrete_sequence=px.colors.sequential.Blues,
                                 height=250) # Ukuran lebih kecil untuk FAQ
            fig_day_faq.update_layout(xaxis_tickangle=-45, margin=dict(l=0, r=0, t=30, b=0)) # Margin agar rapi
            st.plotly_chart(fig_day_faq, use_container_width=True, config={'displayModeBar': False}) # Tanpa mode bar
            peak_day_sales_faq = sales_by_day_faq.loc[sales_by_day_faq['Total Penjualan Bersih (Rp)'].idxmax()]
            st.info(f"👉 Hari Penjualan Puncak: **{peak_day_sales_faq['Hari']}** (Rp {peak_day_sales_faq['Total Penjualan Bersih (Rp)']:.0f})")
            st.caption("Informasi ini dapat membantu perencanaan promosi atau staf.")
            st.markdown("---")
        else:
            st.warning("Tidak dapat menentukan hari penjualan puncak karena data hari dalam seminggu tidak tersedia atau kosong.")



else:
    st.info("⚠️ Silakan sesuaikan filter Anda di sidebar. Tidak ada data yang cocok untuk ditampilkan saat ini.")

