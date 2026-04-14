import streamlit as st
import cv2
import numpy as np
import io
from scipy.spatial import KDTree
from sklearn.cluster import MiniBatchKMeans
from scipy.signal import find_peaks

st.set_page_config(page_title="Grand Carpet AI", layout="wide", page_icon="🧶")

# ═══════════════════════════════════════════════════════════════
# PREMIUM CSS — iOS 17 Light Glassmorphic Design
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Font: Inter for text, but exclude Streamlit icon elements */
    h1, h2, h3, h4, h5, h6, p, span, label, div, input, button, a, li, td, th, textarea, select { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; 
    }
    
    /* Sidebar collapse button: hide broken icon text, show clean symbol */
    [data-testid="stSidebarCollapseButton"] button span {
        font-size: 0 !important;
        line-height: 0 !important;
    }
    [data-testid="stSidebarCollapseButton"] button::after {
        content: "☰";
        font-size: 20px;
        font-family: system-ui !important;
        color: #48484A;
    }
    
    /* Fix upload button text doubling */
    [data-testid="stFileUploader"] button {
        font-size: 0.85rem !important;
        overflow: hidden !important;
    }
    [data-testid="stFileUploader"] section > button > div {
        display: flex !important;
        align-items: center !important;
    }
    
    .stApp { background: #F2F2F7 !important; }
    
    header[data-testid="stHeader"] {
        background: rgba(242,242,247,0.72) !important;
        backdrop-filter: saturate(180%) blur(20px) !important;
        -webkit-backdrop-filter: saturate(180%) blur(20px) !important;
        border-bottom: 0.5px solid rgba(0,0,0,0.06);
    }
    
    section[data-testid="stSidebar"] {
        background: rgba(255,255,255,0.72) !important;
        backdrop-filter: saturate(180%) blur(20px) !important;
        -webkit-backdrop-filter: saturate(180%) blur(20px) !important;
        border-right: 0.5px solid rgba(0,0,0,0.08);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stRadio label span {
        color: #1C1C1E !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        background: rgba(0,0,0,0.03);
        border: 1px solid rgba(0,0,0,0.06);
        border-radius: 12px; padding: 10px 16px; margin: 4px 0;
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1); cursor: pointer;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(212,168,83,0.08);
        border-color: rgba(212,168,83,0.3);
        transform: translateX(4px);
    }
    
    h1 { font-weight: 700 !important; letter-spacing: -0.03em !important; color: #1C1C1E !important; -webkit-text-fill-color: #1C1C1E !important; }
    h2, h3 { font-weight: 600 !important; letter-spacing: -0.02em !important; color: #1C1C1E !important; }
    h3 { font-size: 1.05rem !important; color: #3A3A3C !important; margin-top: 1.2em !important; }
    p, li { color: #48484A !important; line-height: 1.6 !important; }
    span, label { color: #3A3A3C !important; }
    
    .stButton > button, .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #D4A853, #C49640) !important;
        color: #FFF !important; border: none !important; border-radius: 14px !important;
        padding: 12px 32px !important; font-weight: 600 !important; font-size: 0.95rem !important;
        transition: all 0.35s cubic-bezier(0.4,0,0.2,1) !important;
        box-shadow: 0 2px 12px rgba(196,150,64,0.25) !important;
    }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 24px rgba(196,150,64,0.35) !important; }
    .stButton > button:active { transform: translateY(0) scale(0.98) !important; }
    
    .stDownloadButton > button {
        background: rgba(255,255,255,0.8) !important; color: #C49640 !important;
        border: 1px solid rgba(196,150,64,0.3) !important; border-radius: 14px !important;
        padding: 12px 24px !important; font-weight: 600 !important;
        backdrop-filter: blur(10px) !important; transition: all 0.35s cubic-bezier(0.4,0,0.2,1) !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(212,168,83,0.1) !important; border-color: rgba(196,150,64,0.5) !important;
        transform: translateY(-2px) !important; box-shadow: 0 4px 16px rgba(196,150,64,0.15) !important;
    }
    
    .stNumberInput input, .stTextInput input, .stSelectbox select {
        background: rgba(255,255,255,0.8) !important; border: 1px solid rgba(0,0,0,0.08) !important;
        border-radius: 12px !important; color: #1C1C1E !important; padding: 10px 14px !important;
        transition: all 0.3s ease !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus { border-color: rgba(212,168,83,0.5) !important; box-shadow: 0 0 0 3px rgba(212,168,83,0.12) !important; }
    .stNumberInput button { background: rgba(0,0,0,0.04) !important; border: 1px solid rgba(0,0,0,0.08) !important; color: #48484A !important; border-radius: 8px !important; transition: all 0.2s ease !important; }
    .stNumberInput button:hover { background: rgba(212,168,83,0.1) !important; color: #C49640 !important; }
    
    .stSlider [data-testid="stThumbValue"] { color: #C49640 !important; font-weight: 600 !important; }
    
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.6); border: 2px dashed rgba(0,0,0,0.1);
        border-radius: 16px; padding: 20px; backdrop-filter: blur(10px); transition: all 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover { border-color: rgba(212,168,83,0.4); background: rgba(255,255,255,0.8); }
    [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p, [data-testid="stFileUploader"] label { color: #48484A !important; }
    
    .stAlert { border-radius: 14px !important; border: none !important; }
    
    [data-testid="stMetricValue"] { font-weight: 700 !important; color: #C49640 !important; font-size: 1.3rem !important; }
    [data-testid="stMetricLabel"] { font-weight: 500 !important; color: #48484A !important; }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.7); border: 1px solid rgba(0,0,0,0.05);
        border-radius: 16px; padding: 16px; backdrop-filter: blur(10px); transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover { background: rgba(255,255,255,0.9); box-shadow: 0 4px 20px rgba(0,0,0,0.06); transform: translateY(-1px); }
    
    [data-testid="stImage"] { border-radius: 16px; overflow: hidden; box-shadow: 0 4px 24px rgba(0,0,0,0.08); transition: all 0.4s cubic-bezier(0.4,0,0.2,1); }
    [data-testid="stImage"]:hover { box-shadow: 0 8px 40px rgba(0,0,0,0.12); transform: translateY(-2px); }
    [data-testid="stImage"] img { border-radius: 16px; }
    
    [data-testid="stCaptionContainer"] { text-align: center; font-size: 0.82rem !important; margin-top: 8px; }
    [data-testid="stCaptionContainer"] p { color: #8E8E93 !important; }
    
    hr { border: none !important; height: 0.5px !important; background: rgba(0,0,0,0.08) !important; margin: 2em 0 !important; }
    
    .stCheckbox label span { color: #3A3A3C !important; }
    .stCheckbox label:hover span { color: #C49640 !important; }
    .stRadio label span { color: #1C1C1E !important; }
    
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.12); border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.2); }
    
    div[style*="border-radius: 6px"] { border-radius: 12px !important; transition: all 0.3s ease; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    div[style*="border-radius: 6px"]:hover { transform: scale(1.08); box-shadow: 0 4px 16px rgba(0,0,0,0.15); }
    
    @keyframes fadeSlideUp { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
    .main .block-container { animation: fadeSlideUp 0.5s cubic-bezier(0.4,0,0.2,1); }
    
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style="text-align:center; padding: 16px 0 8px 0;">
    <svg width="52" height="52" viewBox="0 0 280 280" xmlns="http://www.w3.org/2000/svg" style="margin-bottom: 6px;">
        <rect x="10" y="10" width="260" height="260" rx="52" ry="52" fill="#1C1C1E"/>
        <text x="140" y="135" text-anchor="middle" fill="white" font-family="Georgia, serif" font-size="72" font-weight="400" letter-spacing="-2">Grand</text>
        <text x="140" y="185" text-anchor="middle" fill="rgba(255,255,255,0.75)" font-family="Helvetica, sans-serif" font-size="42" font-weight="300" letter-spacing="3">carpet</text>
        <rect x="178" y="200" width="58" height="28" rx="8" ry="8" fill="#D4A853"/>
        <text x="207" y="220" text-anchor="middle" fill="#1C1C1E" font-family="sans-serif" font-size="18" font-weight="700" letter-spacing="1">AI</text>
    </svg>
    <div style="font-size: 0.72rem; color: #8E8E93; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px;">Desinatör Otomasyon</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

stage = st.sidebar.radio("Aşama", 
    ("1. Aşama: Renk İndirgeme", 
     "2. Aşama: Pikselleştirme"),
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="text-align:center; padding: 10px 0; opacity: 0.35;">
    <div style="font-size: 0.7rem; color: #8E8E93; letter-spacing: 0.05em;">v2.0 · Grand Carpet AI</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# HERO SECTION — Logo + Branding (Sayfanın başlangıç noktası)
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align: center; padding: 30px 0 10px 0;">
    <svg width="120" height="120" viewBox="0 0 280 280" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <linearGradient id="goldGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#E8C86A"/>
                <stop offset="100%" style="stop-color:#C49640"/>
            </linearGradient>
        </defs>
        <!-- Dark rounded square background -->
        <rect x="10" y="10" width="260" height="260" rx="52" ry="52" fill="#1C1C1E"/>
        <!-- Grand — serif style -->
        <text x="140" y="135" text-anchor="middle" fill="white" font-family="Georgia, 'Times New Roman', serif" font-size="72" font-weight="400" letter-spacing="-2">Grand</text>
        <!-- carpet — light weight -->
        <text x="140" y="185" text-anchor="middle" fill="rgba(255,255,255,0.75)" font-family="'Inter', Helvetica, sans-serif" font-size="42" font-weight="300" letter-spacing="3">carpet</text>
        <!-- AI badge -->
        <rect x="178" y="200" width="58" height="28" rx="8" ry="8" fill="url(#goldGrad)"/>
        <text x="207" y="220" text-anchor="middle" fill="#1C1C1E" font-family="'Inter', sans-serif" font-size="18" font-weight="700" letter-spacing="1">AI</text>
    </svg>
    <div style="font-size: 0.82rem; color: #8E8E93; letter-spacing: 0.04em; margin-top: 14px; font-family: 'Inter', sans-serif;">
        Endüstriyel Halı Desinatör Otomasyon Sistemi
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ----------------- 1. AŞAMA: RENK İNDİRGEME -----------------
if stage == "1. Aşama: Renk İndirgeme":
    st.title("✂️ Renk İndirgeme")
    st.markdown("Halıdaki ışık, gölge ve kumaş dokusunu temizleyerek tasarımı **hedeflenen renk sayısında** net bir dijital çizime dönüştürür.")
    
    uploaded_file = st.file_uploader("Tamamlanmış Fotoğrafı Yükleyin", type=["bmp", "png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Orijinal Fotoğraf", use_container_width=True)
        
        with col2:
            target_colors = st.number_input("Hedef Renk Sayısı (Örn: 8)", min_value=2, max_value=256, value=8)

            smoothing_method = st.radio(
                "Halı Türü ve Tasarım Felsefesi (En Önemli Karar):", 
                (
                    "🏭 Texcelle Endüstriyel (Jakar Örgü ve İlmek Snap)", 
                    "🔺 Düz & Geometrik Halılar (Vektör Etkisi - MeanShift)",
                    "🌪 Kumlu & Kıvrımlı Soyut Halılar (Orijinal Doku - Saf KMeans)"
                )
            )
            
            knot_width = st.number_input("Halı/Desen Yatay İlmek (Knot Grid) Sayısı", min_value=10, max_value=2000, value=200)
            
            # Soyut desenli halılarda ışık eşitleme kasıtlı gradyanları yok eder, varsayılan KAPALI
            if smoothing_method == "🌪 Kumlu & Kıvrımlı Soyut Halılar (Orijinal Doku - Saf KMeans)":
                fix_lighting = st.checkbox("💡 Işık Eşitlemesi (Soyut halılarda KAPALI önerilir)", value=False)
            else:
                fix_lighting = st.checkbox("💡 Işık Eşitlemesi (Zorunlu)", value=True)
            
            if st.button("Hizala ve İşle", type="primary"):
                with st.spinner("Endüstriyel hesaplamalar yapılıyor..."):
                    
                    if fix_lighting:
                        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab_img)
                        illumination = cv2.GaussianBlur(l, (101, 101), 0)
                        l_corrected = np.clip(l.astype(np.float32) - illumination.astype(np.float32) + np.mean(l), 0, 255).astype(np.uint8)
                        img = cv2.cvtColor(cv2.merge((l_corrected, a, b)), cv2.COLOR_LAB2BGR)
                    
                    if smoothing_method == "🏭 Texcelle Endüstriyel (Jakar Örgü ve İlmek Snap)":
                        # 1. Ortogonal Knot (İlmek) Dağılımına Zorla (Superpixels Downsampling)
                        h_orig, w_orig = img.shape[:2]
                        knot_h = int(knot_width * (h_orig / w_orig))
                        
                        # Resmi ilmeklere bölüyoruz (Orthogonal Snapping)
                        knot_img = cv2.resize(img, (knot_width, knot_h), interpolation=cv2.INTER_AREA)
                        
                        # 2. Renk Sayısını (KMeans) İlmek bazında yakala
                        lab = cv2.cvtColor(knot_img, cv2.COLOR_BGR2LAB)
                        pixels = lab.reshape(-1, 3)
                        
                        kmeans = MiniBatchKMeans(n_clusters=target_colors, max_iter=50, batch_size=3072, random_state=42, n_init="auto")
                        kmeans.fit(pixels)
                        
                        # 3. Gerçek Renkli Bayer Ordered Dithering (Jakar Formülü)
                        bayer = np.array([
                            [ 0,  8,  2, 10],
                            [12,  4, 14,  6],
                            [ 3, 11,  1,  9],
                            [15,  7, 13,  5]
                        ]) / 16.0
                        bayer = bayer - 0.5 # -0.5 ile +0.5 arasına çek
                        
                        bayer_tiled = np.tile(bayer, (knot_h // 4 + 1, knot_width // 4 + 1))[:knot_h, :knot_width]
                        
                        # Renklerin daha iyi karışması için L (Aydınlık) kanalına mikro zikzaklar (dither) ekliyoruz
                        spread = 45.0 # Örgü Şiddeti
                        lab_img_float = lab.astype(np.float32)
                        lab_img_float[:, :, 0] += bayer_tiled * spread 
                        lab_img_float = np.clip(lab_img_float, 0, 255)
                        
                        # Zikzak eklenmiş pikselleri Paletteki EN YAKIN (Nearest) Orijinal Renklere Zorla 
                        dithered_pixels = lab_img_float.reshape(-1, 3)
                        tree = KDTree(kmeans.cluster_centers_)
                        _, indices = tree.query(dithered_pixels)
                        
                        final_pixels = kmeans.cluster_centers_[indices]
                        final_pixels = np.clip(final_pixels.astype("uint8"), 0, 255)
                        
                        reduced_lab = final_pixels.reshape((knot_h, knot_width, 3))
                        final_bgr = cv2.cvtColor(reduced_lab, cv2.COLOR_LAB2BGR)
                        
                        # 4. Nearest Neighbor ile Orijinal HD boyuta Upscale (Jilet Keskinliği - Kavis ezilmeden)
                        final_scaled = cv2.resize(final_bgr, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
                        final_rgb = cv2.cvtColor(final_scaled, cv2.COLOR_BGR2RGB)
                        
                        st.session_state['cr_rgb'] = final_rgb
                        st.session_state['cr_bgr'] = final_scaled
                        st.session_state['cr_caption'] = f"Sonuç: Tam {target_colors} Renk Jakar Dithering"
                        
                    elif smoothing_method == "🔺 Düz & Geometrik Halılar (Vektör Etkisi - MeanShift)":
                        # KLASİK MEANSHIFT (Geometrik/Modern halılar için muazzam)
                        flat_img = cv2.pyrMeanShiftFiltering(img, sp=12, sr=35)
                        flat_img = cv2.bilateralFilter(flat_img, 9, 75, 75)
                    
                        lab = cv2.cvtColor(flat_img, cv2.COLOR_BGR2LAB)
                        pixels = lab.reshape(-1, 3)
                        
                        kmeans = MiniBatchKMeans(n_clusters=target_colors, max_iter=50, batch_size=3072, random_state=42, n_init="auto")
                        kmeans.fit(pixels)
                        
                        new_pixels = kmeans.cluster_centers_[kmeans.labels_]
                        new_pixels = np.clip(new_pixels.astype("uint8"), 0, 255)
                        
                        reduced_lab = new_pixels.reshape(lab.shape)
                        final_img = cv2.cvtColor(reduced_lab, cv2.COLOR_LAB2BGR)
                        
                        final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                        st.session_state['cr_rgb'] = final_rgb
                        st.session_state['cr_bgr'] = final_img
                        st.session_state['cr_caption'] = f"Sonuç: Vektörel Sınırlarıyla Tam {target_colors} Renk Geometrik Format"

                    elif smoothing_method == "🌪 Kumlu & Kıvrımlı Soyut Halılar (Orijinal Doku - Saf KMeans)":
                        # SAF KMEANS (Sulu boya çamuru olmadan orijinal kıvrımları ve grenleri noktalarla koruyan model)
                        # Sadece çok ufak bir kamera kumlaması bluru atılır
                        flat_img = cv2.GaussianBlur(img, (3, 3), 0)
                        
                        lab = cv2.cvtColor(flat_img, cv2.COLOR_BGR2LAB)
                        pixels = lab.reshape(-1, 3)
                        
                        kmeans = MiniBatchKMeans(n_clusters=target_colors, max_iter=50, batch_size=3072, random_state=42, n_init="auto")
                        kmeans.fit(pixels)
                        
                        new_pixels = kmeans.cluster_centers_[kmeans.labels_]
                        new_pixels = np.clip(new_pixels.astype("uint8"), 0, 255)
                        
                        reduced_lab = new_pixels.reshape(lab.shape)
                        final_img = cv2.cvtColor(reduced_lab, cv2.COLOR_LAB2BGR)
                        
                        final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
                        st.session_state['cr_rgb'] = final_rgb
                        st.session_state['cr_bgr'] = final_img
                        st.session_state['cr_caption'] = f"Sonuç: Kıvrımları Koruyan Tam {target_colors} Renk Saf Doku"
                    
                    # Palette'i session_state'e kaydet
                    st.session_state['cr_palette'] = kmeans.cluster_centers_
        
        # === SONUÇLARI GÖSTER (session_state — buton bloğu dışında, indirme çalışır) ===
        if 'cr_rgb' in st.session_state:
            st.divider()
            st.image(st.session_state['cr_rgb'], caption=st.session_state.get('cr_caption', 'Sonuç'), use_container_width=True)
            
            # === RENK PALETİNİ EKRANDA GÖSTER ===
            if 'cr_palette' in st.session_state:
                st.markdown("### 🎨 Tasarıma Çıkarılan Renk Paleti")
                pal_lab = st.session_state['cr_palette'].reshape(1, -1, 3).astype(np.uint8)
                pal_rgb = cv2.cvtColor(pal_lab, cv2.COLOR_LAB2RGB)[0]
                
                cols = st.columns(len(pal_rgb))
                for idx, color in enumerate(pal_rgb):
                    hex_color = '#%02x%02x%02x' % (color[0], color[1], color[2])
                    with cols[idx]:
                        st.markdown(f'<div style="background-color: {hex_color}; width: 100%; height: 40px; border-radius: 6px; border: 1px solid #777;"></div>', unsafe_allow_html=True)
                        st.caption(f"{hex_color}")
            
            # === BMP İNDİRME BUTONU (artık buton bloğu dışında — çalışır!) ===
            dl_img = st.session_state['cr_bgr']
            is_success, buffer = cv2.imencode(".bmp", dl_img)
            if is_success:
                io_buf = io.BytesIO(buffer)
                st.download_button("📥 Renk İndirgenmiş Halıyı İndir (.BMP)", data=io_buf, file_name="RENK_INDIRGENME_SONUC.bmp", mime="image/bmp")

# ----------------- 2. AŞAMA: PİKSELLEŞTİRME -----------------
elif stage == "2. Aşama: Pikselleştirme":
    st.title("🧩 Pikselleştirme")
    st.markdown("Renk indirgemeden geçmiş görseli, **tarak ve atkı parametreleriyle** makine ızgarasına dönüştürür.")
    
    uploaded_file = st.file_uploader("Renk İndirgenmiş Görseli Yükleyin", type=["bmp", "png", "jpg", "jpeg"], key="pixel_upload")
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is not None:
            org_h, org_w = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(img_rgb, caption=f"Yüklenen ({org_w}×{org_h})", use_container_width=True)
            
            with col2:
                # ── MAKİNE PARAMETRELERİ ──
                st.markdown("### 🏭 Makine Parametreleri")
                mcol1, mcol2 = st.columns(2)
                with mcol1:
                    tarak = st.number_input("Tarak (adet / 10cm)", min_value=5, max_value=200, value=32, key="tarak",
                                            help="Yatay yönde her 10cm'ye düşen ilmek sayısı")
                with mcol2:
                    atki = st.number_input("Atkı (adet / 10cm)", min_value=5, max_value=200, value=35, key="atki",
                                           help="Dikey yönde her 10cm'ye düşen atkı (sıra) sayısı")
                
                # ── HALI EBATLARI ──
                st.markdown("### 📐 Halı Ebatları")
                ecol1, ecol2 = st.columns(2)
                with ecol1:
                    hali_en = st.number_input("Halı Eni (cm)", min_value=10, max_value=2000, value=160, key="hali_en")
                with ecol2:
                    hali_boy = st.number_input("Halı Boyu (cm)", min_value=10, max_value=2000, value=230, key="hali_boy")
                
                # ── OTOMATİK PİKSEL HESAPLAMA ──
                pixel_w = int(tarak * hali_en * 0.1)
                pixel_h = int(atki * hali_boy * 0.1)
                ilmek_w_mm = 10.0 / tarak * 10  # mm cinsinden (10cm / tarak_sayısı)
                ilmek_h_mm = 10.0 / atki * 10   # mm cinsinden
                toplam_ilmek = pixel_w * pixel_h
                
                st.divider()
                st.markdown("### 📊 Hesaplanan Değerler")
                st.success(f"""
                **Piksel Boyutu:** {pixel_w} × {pixel_h} piksel  
                **Formül:** {tarak}×{hali_en}×0.1 = **{pixel_w}** en  /  {atki}×{hali_boy}×0.1 = **{pixel_h}** boy  
                **İlmek Fiziksel Boyut:** {ilmek_w_mm:.2f}mm × {ilmek_h_mm:.2f}mm  
                **Toplam İlmek:** {toplam_ilmek:,}
                """)
                
                # ── MEVCUT PALET TESPİTİ ──
                # Görseldeki benzersiz renkleri tespit et (KMeans KULLANILMAZ)
                unique_bgr = np.unique(img.reshape(-1, 3), axis=0)
                num_colors = len(unique_bgr)
                
                st.divider()
                st.info(f"🎨 Görselde **{num_colors} benzersiz renk** tespit edildi. Tüm renkler aynen korunacak.")
                
                st.divider()
                st.markdown("### 🧵 Dokuma Dokusu")
                texture_enabled = st.checkbox("Kontrollü Dikey Çizgi Dokusu (Üretim Temsili)", value=True, key="tex_on",
                                              help="Her renk bölgesinin içine deterministik dikey şerit yapısı uygular")
                stripe_period = 3
                if texture_enabled:
                    stripe_period = st.slider("Çizgi Periyodu (sütun)", min_value=2, max_value=8, value=3,
                                             help="Her kaç sütunda bir ikincil renk çizgisi oluşur (2=sık, 6=seyrek)", key="tex_period")
            
            if st.button("🧩 Pikselleştir", type="primary", key="btn_pixel"):
                with st.spinner(f"Endüstriyel piksel mozaiği oluşturuluyor ({pixel_w}×{pixel_h})..."):
                    
                    # ═══════════════════════════════════════════
                    # FAZ 1: MEVCUT PALETTE İLE SEGMENTASYON
                    # ═══════════════════════════════════════════
                    
                    # ADIM 1: Görseldeki mevcut renk paletini al (KMeans YOK)
                    lab_full = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    all_pixels = lab_full.reshape(-1, 3).astype(np.float64)
                    
                    # Mevcut benzersiz renkleri LAB uzayında çıkar
                    unique_lab = np.unique(lab_full.reshape(-1, 3), axis=0).astype(np.float64)
                    palette_lab = unique_lab
                    num_colors = len(palette_lab)
                    
                    # ADIM 2: Her pikseli etiketle + blok-mod küçült
                    tree = KDTree(palette_lab)
                    _, pixel_labels = tree.query(all_pixels)
                    label_map = pixel_labels.reshape(org_h, org_w)
                    
                    grid_j = np.clip(np.arange(org_h) * pixel_h // org_h, 0, pixel_h - 1)
                    grid_i = np.clip(np.arange(org_w) * pixel_w // org_w, 0, pixel_w - 1)
                    grid_jj, grid_ii = np.meshgrid(grid_j, grid_i, indexing='ij')
                    knot_flat = (grid_jj * pixel_w + grid_ii).ravel().astype(np.int64)
                    color_flat = label_map.ravel().astype(np.int64)
                    
                    n_knots = pixel_h * pixel_w
                    combined = knot_flat * num_colors + color_flat
                    hist = np.bincount(combined, minlength=n_knots * num_colors)
                    count_matrix = hist[:n_knots * num_colors].reshape(n_knots, num_colors)
                    
                    small_labels = count_matrix.argmax(axis=1).reshape(pixel_h, pixel_w)
                    
                    # ADIM 3: Agresif sınır temizleme (3 geçiş)
                    # Her geçişte <2 aynı renkli komşusu olan pikseller
                    # dominant komşuyla değiştirilir → pütür/kırıntı/tırtık giderilir
                    for _ in range(3):
                        up    = np.roll(small_labels, 1, axis=0)
                        down  = np.roll(small_labels, -1, axis=0)
                        left  = np.roll(small_labels, 1, axis=1)
                        right = np.roll(small_labels, -1, axis=1)
                        
                        same = ((small_labels == up).astype(np.int32) +
                                (small_labels == down).astype(np.int32) +
                                (small_labels == left).astype(np.int32) +
                                (small_labels == right).astype(np.int32))
                        
                        weak = same < 2
                        weak[0, :] = False; weak[-1, :] = False
                        weak[:, 0] = False; weak[:, -1] = False
                        
                        if not np.any(weak):
                            break
                        
                        nbr = np.zeros((pixel_h, pixel_w, num_colors), dtype=np.int32)
                        for nb in [up, down, left, right]:
                            for c in range(num_colors):
                                nbr[:, :, c] += (nb == c).astype(np.int32)
                        small_labels = np.where(weak, nbr.argmax(axis=-1), small_labels)
                    
                    # Düz versiyon (makine okuma için — 1 piksel = 1 ilmek)
                    flat_lab = palette_lab[small_labels.ravel()].reshape((pixel_h, pixel_w, 3)).astype(np.uint8)
                    snapped_bgr_flat = cv2.cvtColor(flat_lab, cv2.COLOR_LAB2BGR)
                    
                    # ═══════════════════════════════════════════
                    # FAZ 2: KONTROLLÜ İÇ YÜZEY DOKUSU
                    # ═══════════════════════════════════════════
                    
                    if texture_enabled:
                        # ADIM 4a: Her palet rengi için ikincil renk belirle
                        # Paletteki EN YAKIN KOMŞU renk → ikincil
                        p_dists, p_idxs = tree.query(palette_lab, k=2)
                        secondary_palette_idx = p_idxs[:, 1]
                        secondary_palette_dist = p_dists[:, 1]
                        
                        # İkincil renk çok uzaksa → sentetik gölge üret
                        secondary_colors = np.zeros_like(palette_lab)
                        for ci in range(num_colors):
                            if secondary_palette_dist[ci] > 30:
                                shade = palette_lab[ci].copy()
                                shade[0] = max(0, shade[0] - 8)
                                secondary_colors[ci] = shade
                            else:
                                secondary_colors[ci] = palette_lab[secondary_palette_idx[ci]]
                        
                        # ADIM 4b: Deterministik dikey şerit atama
                        # Her rengin fazı farklı → bölgeler arası çizgiler hizalanmaz
                        color_phases = np.arange(num_colors) % stripe_period
                        
                        col_grid = np.broadcast_to(
                            np.arange(pixel_w)[np.newaxis, :], (pixel_h, pixel_w)
                        )
                        knot_phases = color_phases[small_labels.ravel()].reshape(pixel_h, pixel_w)
                        effective_cols = (col_grid + knot_phases) % stripe_period
                        
                        # Şerit maskesi: hangi sütunlar ikincil renk alacak
                        stripe_mask = (effective_cols == 0)
                        
                        # Birincil ve ikincil renk haritaları
                        primary_map = palette_lab[small_labels.ravel()].reshape((pixel_h, pixel_w, 3))
                        secondary_map = secondary_colors[small_labels.ravel()].reshape((pixel_h, pixel_w, 3))
                        
                        # Şerit maskesine göre birleştir
                        output_lab = np.where(
                            stripe_mask[:, :, np.newaxis],
                            secondary_map,
                            primary_map
                        )
                        output_bgr = cv2.cvtColor(output_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
                    else:
                        output_bgr = snapped_bgr_flat.copy()
                    
                    # ═══════════════════════════════════════════
                    # FAZ 3: FİZİKSEL ORANLI RENDER
                    # ═══════════════════════════════════════════
                    
                    # İlmek fiziksel en/boy oranını hesapla
                    # İlmek genişliği = 10mm / tarak, İlmek yüksekliği = 10mm / atkı
                    # Render'da doğru fiziksel oran için dikey çoğaltma/sıkıştırma
                    # Oran = (ilmek_yüksekliği / ilmek_genişliği) = tarak / atkı
                    physical_ratio = tarak / atki  # <1 ise ilmek geniş, >1 ise ilmek uzun
                    
                    # Yatay 1px olarak al, dikeyi oranla
                    # Minimum 1 piksel, en fazla 4 piksel yükseklik
                    if physical_ratio >= 1.0:
                        render_knot_w = 1
                        render_knot_h = max(1, round(physical_ratio))
                    else:
                        # İlmek geniş — yatayda genişlet
                        render_knot_w = max(1, round(1.0 / physical_ratio))
                        render_knot_h = 1
                    
                    # Render — her ilmeği fiziksel oranıyla çoğalt
                    rendered = output_bgr.copy()
                    if render_knot_h > 1:
                        rendered = np.repeat(rendered, render_knot_h, axis=0)
                    if render_knot_w > 1:
                        rendered = np.repeat(rendered, render_knot_w, axis=1)
                    
                    # Sonuçları session_state'e kaydet
                    st.session_state['pxl_rendered'] = rendered
                    st.session_state['pxl_snapped'] = snapped_bgr_flat
                    st.session_state['pxl_palette'] = palette_lab
                    st.session_state['pxl_w'] = pixel_w
                    st.session_state['pxl_h'] = pixel_h
                    st.session_state['pxl_colors'] = len(palette_lab)
                    st.session_state['pxl_knot_w'] = render_knot_w
                    st.session_state['pxl_knot_h'] = render_knot_h
                    st.session_state['pxl_org_w'] = org_w
                    st.session_state['pxl_tarak'] = tarak
                    st.session_state['pxl_atki'] = atki
                    st.session_state['pxl_hali_en'] = hali_en
                    st.session_state['pxl_hali_boy'] = hali_boy
            
            # === SONUÇLARI GÖSTER ===
            if 'pxl_rendered' in st.session_state:
                rendered = st.session_state['pxl_rendered']
                snapped_bgr = st.session_state['pxl_snapped']
                palette_lab = st.session_state['pxl_palette']
                pw = st.session_state['pxl_w']
                ph = st.session_state['pxl_h']
                nc = st.session_state['pxl_colors']
                rkw = st.session_state['pxl_knot_w']
                rkh = st.session_state['pxl_knot_h']
                ow = st.session_state['pxl_org_w']
                sv_tarak = st.session_state['pxl_tarak']
                sv_atki = st.session_state['pxl_atki']
                sv_en = st.session_state['pxl_hali_en']
                sv_boy = st.session_state['pxl_hali_boy']
                
                render_h, render_w = rendered.shape[:2]
                
                st.divider()
                st.markdown("### 📐 Pikselleştirilmiş Sonuç")
                
                # Fiziksel oranla önizleme (halının gerçek en/boy oranıyla göster)
                target_preview_w = min(ow, 1600)
                target_preview_h = int(target_preview_w * (sv_boy / sv_en))
                preview = cv2.resize(rendered, (target_preview_w, target_preview_h), interpolation=cv2.INTER_NEAREST)
                
                preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                st.image(preview_rgb,
                        caption=f"{pw}×{ph} piksel  |  {nc} Renk  |  {sv_tarak}T/{sv_atki}A  |  {sv_en}×{sv_boy}cm",
                        use_container_width=True)
                
                # === ZOOM / DETAY KESİTİ ===
                st.markdown("### 🔍 Piksel Detay (Yakınlaştırma)")
                zcol1, zcol2 = st.columns([1, 2])
                with zcol1:
                    zoom_x = st.slider("Yatay Konum (%)", 0, 100, 15, key="zx")
                    zoom_y = st.slider("Dikey Konum (%)", 0, 100, 15, key="zy")
                    zoom_region = st.slider("Bölge Boyutu (ilmek)", 10, 100, 40, key="zr")
                    show_grid_zoom = st.checkbox("Yakınlaştırmada ızgara göster", value=False, key="grid_zoom")
                
                with zcol2:
                    cx = int(zoom_x / 100 * pw)
                    cy = int(zoom_y / 100 * ph)
                    half = zoom_region // 2
                    
                    x1 = max(0, cx - half)
                    x2 = min(pw, cx + half)
                    y1 = max(0, cy - half)
                    y2 = min(ph, cy + half)
                    
                    # Render koordinatlarına dönüştür
                    rx1, rx2 = x1 * rkw, x2 * rkw
                    ry1, ry2 = y1 * rkh, y2 * rkh
                    
                    crop = rendered[ry1:ry2, rx1:rx2]
                    if crop.size > 0:
                        zoom_scale = max(1, 500 // max(crop.shape[1], 1))
                        zoomed = cv2.resize(crop,
                                           (crop.shape[1] * zoom_scale, crop.shape[0] * zoom_scale),
                                           interpolation=cv2.INTER_NEAREST)
                        
                        # Zoom görünümünde hafif ızgara (opsiyonel)
                        if show_grid_zoom:
                            zs_x = zoom_scale * rkw  # her ilmek render genişliği
                            zs_y = zoom_scale * rkh  # her ilmek render yüksekliği
                            for i in range(1, x2 - x1):
                                gx = i * zs_x
                                if gx < zoomed.shape[1]:
                                    cv2.line(zoomed, (gx, 0), (gx, zoomed.shape[0]), (180, 180, 180), 1)
                            for j in range(1, y2 - y1):
                                gy = j * zs_y
                                if gy < zoomed.shape[0]:
                                    cv2.line(zoomed, (0, gy), (zoomed.shape[1], gy), (180, 180, 180), 1)
                        
                        zoomed_rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
                        st.image(zoomed_rgb,
                                caption=f"Yakınlaştırma: {x2-x1}×{y2-y1} ilmek bölgesi",
                                use_container_width=True)
                
                # === RENK PALETİ ===
                st.markdown("### 🎨 Piksel Paleti")
                pal_show = cv2.cvtColor(palette_lab.reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_LAB2RGB)[0]
                pal_cols = st.columns(len(pal_show))
                for idx, color in enumerate(pal_show):
                    hex_c = '#%02x%02x%02x' % (color[0], color[1], color[2])
                    with pal_cols[idx]:
                        st.markdown(f'<div style="background-color: {hex_c}; width: 100%; height: 40px; border-radius: 6px; border: 1px solid #777;"></div>', unsafe_allow_html=True)
                        st.caption(hex_c)
                
                # === MAKİNE BİLGİ KARTI ===
                st.markdown("### 🏭 Makine Özet")
                info1, info2, info3, info4 = st.columns(4)
                with info1:
                    st.metric("Tarak", f"{sv_tarak}/10cm")
                with info2:
                    st.metric("Atkı", f"{sv_atki}/10cm")
                with info3:
                    st.metric("Halı Ebadı", f"{sv_en}×{sv_boy}cm")
                with info4:
                    st.metric("Toplam İlmek", f"{pw*ph:,}")
                
                # === İNDİRME BUTONLARI ===
                st.markdown("### 📥 İndirme")
                dl1, dl2 = st.columns(2)
                with dl1:
                    ok1, buf1 = cv2.imencode(".bmp", snapped_bgr)
                    if ok1:
                        st.download_button(
                            f"📥 Makine BMP ({pw}×{ph})",
                            data=io.BytesIO(buf1),
                            file_name=f"MAKINE_{sv_tarak}T{sv_atki}A_{pw}x{ph}.bmp",
                            mime="image/bmp", key="dl_raw"
                        )
                        st.caption("1 piksel = 1 ilmek — makine okuma dosyası")
                
                with dl2:
                    ok2, buf2 = cv2.imencode(".bmp", rendered)
                    if ok2:
                        st.download_button(
                            f"📥 Görsel BMP ({render_w}×{render_h})",
                            data=io.BytesIO(buf2),
                            file_name=f"GORSEL_{sv_tarak}T{sv_atki}A_{pw}x{ph}.bmp",
                            mime="image/bmp", key="dl_visual"
                        )
                        st.caption("Fiziksel oranlarıyla render edilmiş görsel")
                
                st.success(f"✅ {sv_tarak}T/{sv_atki}A makine — {sv_en}×{sv_boy}cm halı → {pw}×{ph} = {pw*ph:,} ilmek hazır!")

