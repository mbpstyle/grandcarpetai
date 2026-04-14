import streamlit as st
import cv2
import numpy as np
import io
from scipy.spatial import KDTree
from sklearn.cluster import MiniBatchKMeans
from collections import Counter

st.set_page_config(page_title="Grand Carpet AI", layout="wide", page_icon="🧶")

# ═══════════════════════════════════════════════════════════════
# PREMIUM CSS — iOS 17 Light Glassmorphic Design
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Font: Inter ONLaY for text elements — NOT spans/buttons (breaks Streamlit icons) */
    h1, h2, h3, h4, h5, h6, p, label, div, input, a, li, td, th, textarea, select { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; 
    }
    /* Buttons: apply Inter only to text, not icon spans */
    .stButton button, .stDownloadButton button, .stNumberInput button { 
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; 
    }
    .stMarkdown span, .stRadio span, .stCheckbox span {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Sidebar collapse: hide material icon text completely */
    [data-testid="stSidebarCollapseButton"] button {
        font-size: 0 !important;
        overflow: hidden !important;
        width: 36px !important;
        height: 36px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 10px !important;
        background: rgba(0,0,0,0.04) !important;
        border: none !important;
    }
    [data-testid="stSidebarCollapseButton"] button span {
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        overflow: hidden !important;
        position: absolute !important;
    }
    [data-testid="stSidebarCollapseButton"] button::before {
        content: "☰";
        visibility: visible !important;
        font-size: 18px;
        font-family: system-ui, sans-serif !important;
        color: #48484A;
    }
    
    /* File uploader: fix doubled text */
    [data-testid="stFileUploader"] button span[data-testid="stMarkdownContainer"],
    [data-testid="stFileUploader"] button > div > span:first-child {
        font-family: 'Inter', sans-serif !important;
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
    st.title("🎨 Renk İndirgeme")
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

            fix_lighting = st.checkbox("💡 Işık Eşitlemesi (Fotoğraftaki gölge/ışık farkını dengeler)", value=True)

            if st.button("Hizala ve İşle", type="primary"):
                with st.spinner("Renk indirgeme yapılıyor..."):

                    # Opsiyonel ışık eşitleme (LAB L kanalı üzerinden)
                    if fix_lighting:
                        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab_img)
                        illumination = cv2.GaussianBlur(l, (101, 101), 0)
                        l_corrected = np.clip(l.astype(np.float32) - illumination.astype(np.float32) + np.mean(l), 0, 255).astype(np.uint8)
                        img = cv2.cvtColor(cv2.merge((l_corrected, a, b)), cv2.COLOR_LAB2BGR)

                    # Hafif gürültü temizleme (kamera kumlaması)
                    img_clean = cv2.GaussianBlur(img, (3, 3), 0)

                    # LAB renk uzayında K-Means (tam çözünürlükte, ön küçültme yok)
                    lab = cv2.cvtColor(img_clean, cv2.COLOR_BGR2LAB)
                    pixels = lab.reshape(-1, 3).astype(np.float32)

                    kmeans = MiniBatchKMeans(
                        n_clusters=target_colors, max_iter=50,
                        batch_size=3072, random_state=42, n_init="auto"
                    )
                    kmeans.fit(pixels)

                    # Her pikseli en yakın palet rengine eşle (nearest neighbor — renk karışımı yok)
                    new_pixels = kmeans.cluster_centers_[kmeans.labels_]
                    new_pixels = np.clip(new_pixels.astype("uint8"), 0, 255)

                    reduced_lab = new_pixels.reshape(lab.shape)
                    final_img = cv2.cvtColor(reduced_lab, cv2.COLOR_LAB2BGR)
                    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

                    st.session_state['cr_rgb'] = final_rgb
                    st.session_state['cr_bgr'] = final_img
                    st.session_state['cr_caption'] = f"Sonuç: {target_colors} Renk İndirgeme"
                    st.session_state['cr_palette'] = kmeans.cluster_centers_
                    st.session_state['cr_labels'] = kmeans.labels_
                    st.session_state['cr_total_pixels'] = len(kmeans.labels_)

        # === SONUÇLARI GÖSTER (session_state — buton bloğu dışında, indirme çalışır) ===
        if 'cr_rgb' in st.session_state:
            st.divider()
            st.image(st.session_state['cr_rgb'], caption=st.session_state.get('cr_caption', 'Sonuç'), use_container_width=True)

            # === RENK PALETİNİ EKRANDA GÖSTER (sıklık bilgisiyle) ===
            if 'cr_palette' in st.session_state:
                st.markdown("### 🎨 Tasarıma Çıkarılan Renk Paleti")
                pal_lab = st.session_state['cr_palette'].reshape(1, -1, 3).astype(np.uint8)
                pal_rgb = cv2.cvtColor(pal_lab, cv2.COLOR_LAB2RGB)[0]

                # Sıklık hesapla
                label_counts = Counter(st.session_state['cr_labels'])
                total_px = st.session_state['cr_total_pixels']

                # Sıklığa göre sırala (en yaygın renk solda)
                sorted_indices = sorted(label_counts.keys(), key=lambda k: label_counts[k], reverse=True)

                cols = st.columns(len(pal_rgb))
                for col_idx, pal_idx in enumerate(sorted_indices):
                    color = pal_rgb[pal_idx]
                    hex_color = '#%02x%02x%02x' % (color[0], color[1], color[2])
                    count = label_counts[pal_idx]
                    pct = (count / total_px) * 100
                    with cols[col_idx]:
                        st.markdown(f'<div style="background-color: {hex_color}; width: 100%; height: 40px; border-radius: 6px; border: 1px solid #777;"></div>', unsafe_allow_html=True)
                        st.caption(f"{hex_color}")
                        st.caption(f"%{pct:.1f}")

            # === BMP İNDİRME BUTONU ===
            dl_img = st.session_state['cr_bgr']
            is_success, buffer = cv2.imencode(".bmp", dl_img)
            if is_success:
                io_buf = io.BytesIO(buffer)
                st.download_button("📥 Renk İndirgenmiş Halıyı İndir (.BMP)", data=io_buf, file_name="RENK_INDIRGENME_SONUC.bmp", mime="image/bmp")

# ----------------- 2. AŞAMA: PİKSELLEŞTİRME -----------------
elif stage == "2. Aşama: Pikselleştirme":
    st.title("🔲 Pikselleştirme")
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
                    # FAZ 1: PALETİ ÇIKAR + LABEL HARİTASI
                    # ═══════════════════════════════════════════

                    lab_full = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    unique_bgr_pre = np.unique(img.reshape(-1, 3), axis=0)
                    num_raw = len(unique_bgr_pre)

                    if num_raw <= 20:
                        # Az renk (Stage 1 çıktısı veya hazır desen) → doğrudan kullan
                        palette_lab = np.unique(
                            lab_full.reshape(-1, 3), axis=0
                        ).astype(np.float64)
                        tree = KDTree(palette_lab)
                        _, flat_lbl = tree.query(lab_full.reshape(-1, 3).astype(np.float64))
                        label_map = flat_lbl.reshape(org_h, org_w).astype(np.uint8)
                    else:
                        # Çok fazla renk (JPEG/AA gürültüsü) → otomatik KMeans indirgeme
                        n_auto = min(num_raw, 12)
                        km_pre = MiniBatchKMeans(
                            n_clusters=n_auto, n_init="auto",
                            random_state=42, max_iter=100
                        )
                        km_pre.fit(lab_full.reshape(-1, 3).astype(np.float32))
                        palette_lab = km_pre.cluster_centers_.astype(np.float64)
                        tree = KDTree(palette_lab)
                        label_map = km_pre.labels_.reshape(org_h, org_w).astype(np.uint8)

                    num_colors = len(palette_lab)

                    # ═══════════════════════════════════════════
                    # FAZ 2: LABEL HARİTASINI ÖLÇEKLE
                    # ═══════════════════════════════════════════
                    # INTER_NEAREST label map üzerinde çalışır:
                    #   - Büyütme + küçültme her ikisinde de boşluk bırakmaz
                    #   - Renk interpolasyonu olmaz, label değerleri karışmaz
                    #   - Önceki majority vote kodu büyütmede bazı hedef sütunlara
                    #     hiç kaynak piksel map etmiyordu → o sütunlar hep 0.renk
                    #     (lacivert) çıkıyordu → şekiller bozuluyordu. Bu düzeltir.
                    small_labels = cv2.resize(
                        label_map, (pixel_w, pixel_h),
                        interpolation=cv2.INTER_NEAREST
                    ).astype(np.int32)

                    # ═══════════════════════════════════════════
                    # FAZ 3: MİNİMAL TEMİZLEME (2 geçiş)
                    # ═══════════════════════════════════════════
                    # Sadece tamamen izole pikselleri (0 aynı-renkli komşu) temizle.
                    # Diagonal kenar piksellerinin 1 komşusu olduğu için onlara dokunulmaz
                    # → köşelerde girdili-çıktılı etki oluşmaz.
                    for _ in range(2):
                        up    = np.roll(small_labels, 1, axis=0)
                        down  = np.roll(small_labels, -1, axis=0)
                        left  = np.roll(small_labels, 1, axis=1)
                        right = np.roll(small_labels, -1, axis=1)

                        same = ((small_labels == up).astype(np.int32) +
                                (small_labels == down).astype(np.int32) +
                                (small_labels == left).astype(np.int32) +
                                (small_labels == right).astype(np.int32))

                        weak = same == 0
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
