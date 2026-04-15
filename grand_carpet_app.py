import streamlit as st
import cv2
import numpy as np
import io
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

        # Orijinal renk sayısını tespit et
        unique_orig = len(np.unique(img_rgb.reshape(-1, 3), axis=0))

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Orijinal Fotoğraf", use_container_width=True)
            st.info(f"🎨 Orijinal görselde **{unique_orig:,}** benzersiz renk tespit edildi.")

        with col2:
            target_colors = st.number_input("Hedef Renk Sayısı (Örn: 8)", min_value=2, max_value=256, value=8)

            fix_lighting = st.checkbox("💡 Işık Eşitlemesi (Fotoğraftaki gölge/ışık farkını dengeler)", value=False)

            if st.button("Hizala ve İşle", type="primary"):
                with st.spinner("Renk indirgeme yapılıyor..."):

                    # Opsiyonel ışık eşitleme (LAB L kanalı üzerinden)
                    if fix_lighting:
                        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab_img)
                        illumination = cv2.GaussianBlur(l, (101, 101), 0)
                        l_corrected = np.clip(l.astype(np.float32) - illumination.astype(np.float32) + np.mean(l), 0, 255).astype(np.uint8)
                        img = cv2.cvtColor(cv2.merge((l_corrected, a, b)), cv2.COLOR_LAB2BGR)

                    # ═══════════════════════════════════════════
                    # HTML BİREBİR AYNI: RGB K-Means
                    # Blur yok, LAB dönüşümü yok, örnekleme var
                    # ═══════════════════════════════════════════
                    img_rgb_work = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h_img, w_img = img_rgb_work.shape[:2]

                    # Tüm pikselleri RGB olarak al
                    all_pixels = img_rgb_work.reshape(-1, 3).astype(np.float64)

                    # HTML'deki gibi ~6000 piksel örnekle
                    step = max(1, len(all_pixels) // 6000)
                    sample = all_pixels[::step]

                    # Saf K-Means (HTML'deki kMeans fonksiyonunun aynısı)
                    k = target_colors
                    max_iters = 25

                    # Rastgele merkez başlat
                    rng = np.random.default_rng(42)
                    indices = rng.choice(len(sample), size=k, replace=False)
                    centers = sample[indices].copy()

                    for it in range(max_iters):
                        # Her örnek pikseli en yakın merkeze ata
                        dists = np.linalg.norm(sample[:, None, :] - centers[None, :, :], axis=2)
                        labels = np.argmin(dists, axis=1)

                        # Yeni merkezleri hesapla
                        new_centers = np.zeros_like(centers)
                        changed = False
                        for ci in range(k):
                            mask = labels == ci
                            if np.any(mask):
                                nc = np.round(sample[mask].mean(axis=0))
                                if not np.array_equal(nc, centers[ci]):
                                    changed = True
                                new_centers[ci] = nc
                            else:
                                new_centers[ci] = centers[ci]

                        centers = new_centers
                        if not changed:
                            break

                    # Tüm pikselleri palette eşle (HTML'deki mapToPalette — chunk'lı bellek güvenli)
                    chunk_size = 100000
                    all_labels = np.zeros(len(all_pixels), dtype=np.int32)
                    for start in range(0, len(all_pixels), chunk_size):
                        end = min(start + chunk_size, len(all_pixels))
                        chunk = all_pixels[start:end]
                        dists = np.linalg.norm(chunk[:, None, :] - centers[None, :, :], axis=2)
                        all_labels[start:end] = np.argmin(dists, axis=1)
                    mapped = centers[all_labels].astype(np.uint8)

                    final_rgb = mapped.reshape(h_img, w_img, 3)
                    final_bgr = cv2.cvtColor(final_rgb, cv2.COLOR_RGB2BGR)

                    st.session_state['cr_rgb'] = final_rgb
                    st.session_state['cr_bgr'] = final_bgr
                    st.session_state['cr_caption'] = f"Sonuç: {target_colors} Renk İndirgeme"
                    st.session_state['cr_palette_rgb'] = centers
                    st.session_state['cr_labels'] = all_labels
                    st.session_state['cr_total_pixels'] = len(all_labels)

        # === SONUÇLARI GÖSTER (session_state — buton bloğu dışında, indirme çalışır) ===
        if 'cr_rgb' in st.session_state:
            st.divider()
            st.image(st.session_state['cr_rgb'], caption=st.session_state.get('cr_caption', 'Sonuç'), use_container_width=True)

            # === RENK PALETİNİ EKRANDA GÖSTER (sıklık bilgisiyle) ===
            if 'cr_palette_rgb' in st.session_state:
                st.markdown("### 🎨 Tasarıma Çıkarılan Renk Paleti")
                pal_rgb = st.session_state['cr_palette_rgb'].astype(np.uint8)

                # Sıklık hesapla
                label_counts = Counter(st.session_state['cr_labels'])
                total_px = st.session_state['cr_total_pixels']

                # Sıklığa göre sırala (en yaygın renk solda)
                sorted_indices = sorted(label_counts.keys(), key=lambda k: label_counts[k], reverse=True)

                cols = st.columns(len(pal_rgb))
                for col_idx, pal_idx in enumerate(sorted_indices):
                    color = pal_rgb[pal_idx]
                    hex_color = '#%02x%02x%02x' % (int(color[0]), int(color[1]), int(color[2]))
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

    # Kaynak: session_state'deki renk indirgenmiş görsel VEYA yeni yükleme
    source_from_session = 'cr_bgr' in st.session_state
    if source_from_session:
        st.info("✅ 1. Aşamadan renk indirgenmiş görsel algılandı. Doğrudan kullanılacak (BMP kayıp yok).")

    uploaded_file = st.file_uploader("Veya farklı bir görsel yükleyin", type=["bmp", "png", "jpg", "jpeg"], key="pixel_upload")

    # Kaynak belirleme: session > upload
    img = None
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    elif source_from_session:
        img = st.session_state['cr_bgr']

    if img is not None:
        org_h, org_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Yüklenen görseldeki renk sayısını tespit et
        unique_pix = len(np.unique(img.reshape(-1, 3), axis=0))

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption=f"Yüklenen ({org_w}×{org_h})", use_container_width=True)
            st.info(f"🎨 Görselde **{unique_pix:,}** benzersiz renk tespit edildi.")

        with col2:
            st.markdown("### 🏭 Makine Parametreleri")
            mcol1, mcol2 = st.columns(2)
            with mcol1:
                tarak = st.number_input("Tarak (adet / 10cm)", min_value=5, max_value=200, value=32, key="tarak",
                                        help="Yatay yönde her 10cm'ye düşen ilmek sayısı")
            with mcol2:
                atki = st.number_input("Atkı (adet / 10cm)", min_value=5, max_value=200, value=35, key="atki",
                                       help="Dikey yönde her 10cm'ye düşen atkı (sıra) sayısı")

            st.markdown("### 📐 Halı Ebatları")
            ecol1, ecol2 = st.columns(2)
            with ecol1:
                hali_en = st.number_input("Halı Eni (cm)", min_value=10, max_value=2000, value=160, key="hali_en")
            with ecol2:
                hali_boy = st.number_input("Halı Boyu (cm)", min_value=10, max_value=2000, value=230, key="hali_boy")

            # Hesaplama: tarak ve atkı 10cm'deki diş/ilmek sayısı
            # En pikseli = (tarak / 10) × halı_eni_cm = tarak × halı_eni / 10
            # Boy pikseli = (atkı / 10) × halı_boyu_cm = atkı × halı_boyu / 10
            tarak_per_cm = tarak / 10.0
            atki_per_cm = atki / 10.0
            pixel_w = round(tarak_per_cm * hali_en)
            pixel_h = round(atki_per_cm * hali_boy)

            # 1 ilmeğin fiziksel boyutu
            ilmek_w_mm = 10.0 / tarak   # cm → mm için ×10 gerek yok, 10cm/tarak = cm, ama mm istiyoruz
            ilmek_w_mm = 100.0 / tarak  # 100mm / tarak = mm cinsinden 1 ilmek genişliği
            ilmek_h_mm = 100.0 / atki   # 100mm / atkı = mm cinsinden 1 ilmek yüksekliği

            toplam_ilmek = pixel_w * pixel_h

            # Metrekareye düşen ilmek: (tarak_per_m) × (atkı_per_m)
            tarak_per_m = tarak * 10   # 10cm'de tarak → 100cm'de tarak×10
            atki_per_m = atki * 10
            ilmek_per_m2 = tarak_per_m * atki_per_m

            # Halı alanı
            hali_alan_m2 = (hali_en / 100.0) * (hali_boy / 100.0)

            st.divider()
            st.markdown("### 📊 Hesaplanan Değerler")
            st.success(f"""
**Piksel Boyutu:** {pixel_w} × {pixel_h} piksel
**Formül:** {tarak_per_cm:.1f} ilmek/cm × {hali_en} cm = **{pixel_w}** en  /  {atki_per_cm:.1f} ilmek/cm × {hali_boy} cm = **{pixel_h}** boy
**1 İlmek Boyutu:** {ilmek_w_mm:.2f}mm × {ilmek_h_mm:.2f}mm
**Metrekare Yoğunluk:** {ilmek_per_m2:,} ilmek/m²
**Halı Alanı:** {hali_alan_m2:.2f} m²
**Toplam İlmek:** {toplam_ilmek:,}
            """)

        if st.button("🧩 Pikselleştir", type="primary", key="btn_pixel"):
            with st.spinner(f"Pikselleştirme yapılıyor ({pixel_w}×{pixel_h})..."):

                # HTML birebir aynı: Math.floor(y * sH / H) nearest-neighbor
                src_rows = (np.arange(pixel_h) * org_h // pixel_h).astype(int)
                src_cols = (np.arange(pixel_w) * org_w // pixel_w).astype(int)
                out = img[np.ix_(src_rows, src_cols)]

                st.session_state['pxl_result'] = out
                st.session_state['pxl_source_img'] = img.copy()   # karşılaştırma için
                st.session_state['pxl_w'] = pixel_w
                st.session_state['pxl_h'] = pixel_h
                st.session_state['pxl_org_w'] = org_w
                st.session_state['pxl_tarak'] = tarak
                st.session_state['pxl_atki'] = atki
                st.session_state['pxl_hali_en'] = hali_en
                st.session_state['pxl_hali_boy'] = hali_boy
                st.session_state.pop('heat_img', None)  # yeni piksel → eski haritayı temizle

        # === SONUÇLARI GÖSTER ===
        if 'pxl_result' in st.session_state:
            result_bgr = st.session_state['pxl_result']
            pw = st.session_state['pxl_w']
            ph = st.session_state['pxl_h']
            ow = st.session_state['pxl_org_w']
            sv_tarak = st.session_state['pxl_tarak']
            sv_atki = st.session_state['pxl_atki']
            sv_en = st.session_state['pxl_hali_en']
            sv_boy = st.session_state['pxl_hali_boy']

            unique_colors = len(np.unique(result_bgr.reshape(-1, 3), axis=0))

            st.divider()
            st.markdown("### 📐 Pikselleştirilmiş Sonuç")

            target_preview_w = min(ow, 1600)
            target_preview_h = int(target_preview_w * (sv_boy / sv_en))
            preview = cv2.resize(result_bgr, (target_preview_w, target_preview_h), interpolation=cv2.INTER_NEAREST)

            preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            st.image(preview_rgb,
                    caption=f"{pw}×{ph} piksel  |  {unique_colors} Renk  |  {sv_tarak}T/{sv_atki}A  |  {sv_en}×{sv_boy}cm",
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

                crop = result_bgr[y1:y2, x1:x2]
                if crop.size > 0:
                    zoom_scale = max(1, 500 // max(crop.shape[1], 1))
                    zoomed = cv2.resize(crop,
                                       (crop.shape[1] * zoom_scale, crop.shape[0] * zoom_scale),
                                       interpolation=cv2.INTER_NEAREST)

                    if show_grid_zoom:
                        for i in range(1, x2 - x1):
                            gx = i * zoom_scale
                            if gx < zoomed.shape[1]:
                                cv2.line(zoomed, (gx, 0), (gx, zoomed.shape[0]), (180, 180, 180), 1)
                        for j in range(1, y2 - y1):
                            gy = j * zoom_scale
                            if gy < zoomed.shape[0]:
                                cv2.line(zoomed, (0, gy), (zoomed.shape[1], gy), (180, 180, 180), 1)

                    zoomed_rgb = cv2.cvtColor(zoomed, cv2.COLOR_BGR2RGB)
                    st.image(zoomed_rgb,
                            caption=f"Yakınlaştırma: {x2-x1}×{y2-y1} ilmek bölgesi",
                            use_container_width=True)

            # === RENK PALETİ ===
            st.markdown("### 🎨 Piksel Paleti")
            pixel_tuples = [tuple(p) for p in result_bgr.reshape(-1, 3)]
            color_counts = Counter(pixel_tuples)
            sorted_colors = color_counts.most_common()
            total_px = pw * ph

            show_colors = sorted_colors[:16]
            pal_cols = st.columns(min(len(show_colors), 8))
            for idx, (bgr_tuple, count) in enumerate(show_colors):
                col_idx = idx % min(len(show_colors), 8)
                rgb = (bgr_tuple[2], bgr_tuple[1], bgr_tuple[0])
                hex_c = '#%02x%02x%02x' % rgb
                pct = (count / total_px) * 100
                with pal_cols[col_idx]:
                    st.markdown(f'<div style="background-color: {hex_c}; width: 100%; height: 40px; border-radius: 6px; border: 1px solid #777;"></div>', unsafe_allow_html=True)
                    st.caption(f"{hex_c}")
                    st.caption(f"%{pct:.1f}")

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
                ok1, buf1 = cv2.imencode(".bmp", result_bgr)
                if ok1:
                    st.download_button(
                        f"📥 Makine BMP ({pw}×{ph})",
                        data=io.BytesIO(buf1),
                        file_name=f"MAKINE_{sv_tarak}T{sv_atki}A_{pw}x{ph}.bmp",
                        mime="image/bmp", key="dl_raw"
                    )
                    st.caption("1 piksel = 1 ilmek — makine okuma dosyası")

            with dl2:
                ok2, buf2 = cv2.imencode(".png", preview)
                if ok2:
                    st.download_button(
                        f"📥 Önizleme PNG ({target_preview_w}×{target_preview_h})",
                        data=io.BytesIO(buf2),
                        file_name=f"ONIZLEME_{sv_tarak}T{sv_atki}A_{pw}x{ph}.png",
                        mime="image/png", key="dl_visual"
                    )
                    st.caption("Fiziksel oranlarıyla büyütülmüş önizleme")

            st.success(f"✅ {sv_tarak}T/{sv_atki}A makine — {sv_en}×{sv_boy}cm halı → {pw}×{ph} = {pw*ph:,} ilmek hazır!")

            st.markdown("---")

            # ═══════════════════════════════════════════════
            # A: YAN YANA KARŞILAŞTIRMA
            # ═══════════════════════════════════════════════
            st.markdown("### 🔄 Yan Yana Karşılaştırma")
            st.markdown("Orijinal kaynak ile pikselleştirilmiş sonucu aynı fiziksel boyuta getirerek karşılaştırır.")

            # Her iki görseli aynı görsel boyuta getir (orijinalin boyutunda)
            orig_bgr = st.session_state.get('pxl_source_img')
            if orig_bgr is not None:
                # Ortak hedef boyut: max 800px genişlik, orijinal orana göre yükseklik
                comp_w = min(orig_bgr.shape[1], 800)
                comp_h = int(comp_w * orig_bgr.shape[0] / orig_bgr.shape[1])

                orig_resized = cv2.resize(orig_bgr, (comp_w, comp_h), interpolation=cv2.INTER_AREA)
                pxl_resized  = cv2.resize(result_bgr, (comp_w, comp_h), interpolation=cv2.INTER_NEAREST)

                cA, cB = st.columns(2)
                with cA:
                    st.markdown("**Orijinal Kaynak**")
                    st.image(cv2.cvtColor(orig_resized, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.caption(f"{orig_bgr.shape[1]}×{orig_bgr.shape[0]} px")
                with cB:
                    st.markdown("**Pikselleştirilmiş**")
                    st.image(cv2.cvtColor(pxl_resized, cv2.COLOR_BGR2RGB), use_container_width=True)
                    st.caption(f"{pw}×{ph} px  |  {unique_colors} renk")
            else:
                st.info("Karşılaştırma için kaynak görsel session'da bulunamadı. Sayfayı yeniden yükleyin.")

            st.markdown("---")

            # ═══════════════════════════════════════════════
            # B: KABARTMA ISI HARİTASI
            # ═══════════════════════════════════════════════
            st.markdown("### 🌡️ Kabartma Isı Haritası")
            st.markdown("Piksellerin parlaklık ve doygunluk yoğunluğuna göre halıdaki kabartma bölgelerini tahmin eder.")

            blur_r = st.slider("Bulanıklık Yarıçapı", 1, 20, 6, key="heat_blur")

            if st.button("🌡️ Haritayı Oluştur", key="btn_heat"):
                with st.spinner("Isı haritası hesaplanıyor..."):
                    heat_src = result_bgr.astype(np.float32)

                    # Luminance ve saturation hesapla (HTML ile aynı formül)
                    r = heat_src[:, :, 2] / 255.0
                    g = heat_src[:, :, 1] / 255.0
                    b = heat_src[:, :, 0] / 255.0

                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    mx  = np.maximum(np.maximum(r, g), b)
                    mn  = np.minimum(np.minimum(r, g), b)
                    sat = np.where(mx == 0, 0.0, (mx - mn) / mx)

                    intensity = (1.0 - lum) * 0.55 + sat * 0.45

                    # Box blur (HTML'deki boxBlur ile aynı)
                    ksize = blur_r * 2 + 1
                    blurred = cv2.blur(intensity, (ksize, ksize))

                    # Normalize [0, 1]
                    lo, hi = blurred.min(), blurred.max()
                    t = (blurred - lo) / (hi - lo + 1e-9)

                    # Renk skalası (HTML heatColor ile aynı)
                    # stops: koyu mavi → yeşil → turuncu → kırmızı → beyaz
                    stops = np.array([
                        [4,  44,  83],
                        [29, 158, 117],
                        [239, 159, 39],
                        [226, 75,  74],
                        [255, 255, 255]
                    ], dtype=np.float32)

                    n = len(stops) - 1
                    idx = np.clip((t * n).astype(int), 0, n - 1)
                    frac = t * n - idx
                    frac = np.clip(frac, 0, 1)[:, :, np.newaxis]

                    c_lo = stops[idx]
                    c_hi = stops[np.clip(idx + 1, 0, n)]
                    heat_rgb = (c_lo + (c_hi - c_lo) * frac).astype(np.uint8)

                    st.session_state['heat_img'] = heat_rgb

            if 'heat_img' in st.session_state:
                heat_img = st.session_state['heat_img']
                st.image(heat_img, caption="Isı haritası — Koyu mavi: düşük kabartma → Beyaz: yüksek kabartma", use_container_width=True)

                # Renk skalası göster
                gradient_html = """
                <div style="display:flex;align-items:center;gap:10px;margin:8px 0;">
                  <span style="font-size:12px;color:#8E8E93;">Düşük kabartma</span>
                  <div style="flex:1;height:14px;border-radius:6px;background:linear-gradient(to right,#042C53,#1D9E75,#EF9F27,#E24B4A,#ffffff);border:1px solid rgba(0,0,0,0.1);"></div>
                  <span style="font-size:12px;color:#8E8E93;">Yüksek kabartma</span>
                </div>
                """
                st.markdown(gradient_html, unsafe_allow_html=True)

                # İndir
                ok_h, buf_h = cv2.imencode(".png", cv2.cvtColor(heat_img, cv2.COLOR_RGB2BGR))
                if ok_h:
                    st.download_button(
                        "📥 Isı Haritasını İndir (PNG)",
                        data=io.BytesIO(buf_h),
                        file_name=f"ISI_HARITASI_{pw}x{ph}.png",
                        mime="image/png", key="dl_heat"
                    )
