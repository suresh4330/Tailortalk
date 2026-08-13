import streamlit as st
import os
from src.embeddings import EmbeddingGenerator
from src.vector_store import SareeVectorStore
from src.search import VisualSearcher
from src.tools import search_similar_sarees, set_searcher
from src.agent import create_tailortalk_agent
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="TailorTalk", page_icon="🥻", layout="wide")

# ==========================
# CUSTOM CSS FOR EXACT MATCH
# ==========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

header {visibility: hidden;}
footer {visibility: hidden;}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: #111111 !important;
}

h1 {
    font-size: 3.2rem !important;
    line-height: 1.1 !important;
    margin-bottom: 1rem !important;
}

/* Custom "Tabs" text */
.fake-tabs {
    font-size: 0.9rem;
    font-weight: 600;
    color: #111;
    border-bottom: 1px solid #E0E0E0;
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
}
.fake-tabs span {
    margin-right: 2rem;
    cursor: pointer;
}
.fake-tabs .active {
    border-bottom: 2px solid #111;
    padding-bottom: 0.5rem;
}

/* Analyze Button */
.analyze-btn > button {
    background-color: #111111 !important;
    color: #FFFFFF !important;
    border-radius: 4px !important;
    height: 42px !important;
    margin-top: 28px !important;
    width: 100% !important;
}

/* Suggestion Buttons */
.sugg-btn > button {
    background-color: #FFFFFF !important;
    color: #111111 !important;
    border: 1px solid #E0E0E0 !important;
    border-radius: 4px !important;
    width: 100% !important;
    font-weight: 500 !important;
    padding: 0.5rem !important;
}
.sugg-btn > button:hover {
    border-color: #111111 !important;
}

/* Right Column Image Constraint */
.right-image-container {
    background-color: #FFFFFF;
    padding: 1rem;
    border-radius: 4px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    text-align: left;
    max-width: 450px;
}
.right-image-container img {
    max-height: 500px;
    object-fit: cover;
    border-radius: 4px;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# INITIALIZATION
# ==========================
@st.cache_resource(show_spinner="Loading AI Models...")
def init_system():
    emb_gen = EmbeddingGenerator()
    vector_store = SareeVectorStore(dimension=512)
    if os.path.exists('embeddings/faiss.index') and os.path.exists('embeddings/index_meta.json'):
        vector_store.load('embeddings/faiss.index', 'embeddings/index_meta.json')
    searcher = VisualSearcher(emb_gen, vector_store, alpha=0.6, beta=0.4)
    set_searcher(searcher)
    agent = create_tailortalk_agent(tools=[search_similar_sarees])
    return searcher, agent

try:
    searcher, agent = init_system()
except Exception as e:
    st.error(f"Failed to init system: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_image_path" not in st.session_state:
    st.session_state.query_image_path = None
if "raw_results" not in st.session_state:
    st.session_state.raw_results = None

# Top Nav
st.markdown("""
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 1rem 0 3rem 0;">
        <h2 style="margin: 0; font-size: 2rem;">TailorTalk</h2>
        <div style="font-size: 0.9rem; font-weight: 600; color: #555;">
            <span style="border-bottom: 2px solid #111; color: #111; margin-right: 2rem; cursor: pointer;">New Search</span>
            <span style="cursor: pointer;">About</span>
        </div>
    </div>
""", unsafe_allow_html=True)

col_left, col_pad, col_right = st.columns([1.2, 0.1, 1])

trigger_prompt = None

with col_left:
    st.markdown("<h1>Find sarees that look like your style.</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 1.1rem; color: #555; margin-bottom: 2rem;'>Upload an inspiration image or provide a link, and our AI will curate a selection of luxury sarees matching your aesthetic, color palette, and intricate details.</p>", unsafe_allow_html=True)
    
    # Fake tabs exactly like mockup
    st.markdown("<div class='fake-tabs'><span class='active'>Upload Image</span><span>Image URL</span></div>", unsafe_allow_html=True)
    
    # File Uploader
    with st.container():
        uploaded_file = st.file_uploader("Drag and drop an image here, or click to browse (JPG, PNG or WEBP)", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
        if uploaded_file is not None:
            os.makedirs("temp", exist_ok=True)
            temp_path = os.path.join("temp", uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.query_image_path = temp_path
            st.success("Image uploaded successfully!")

    # OR divider
    st.markdown("<div style='text-align: center; color: #999; margin: 1rem 0; font-size: 0.8rem; font-weight: 600;'>OR</div>", unsafe_allow_html=True)
    
    # URL Input + Analyze Button
    c_input, c_btn = st.columns([4, 1])
    with c_input:
        image_url = st.text_input("Paste Image Link", placeholder="https://example.com/image.jpg", label_visibility="collapsed")
    with c_btn:
        st.markdown("<div class='analyze-btn'>", unsafe_allow_html=True)
        if st.button("Analyze", use_container_width=True):
            if image_url:
                import requests
                try:
                    response = requests.get(image_url, timeout=5)
                    response.raise_for_status()
                    os.makedirs("temp", exist_ok=True)
                    temp_path = os.path.join("temp", "url_image.jpg")
                    with open(temp_path, "wb") as f:
                        f.write(response.content)
                    st.session_state.query_image_path = temp_path
                    st.success("URL loaded!")
                    trigger_prompt = "Find sarees similar to this" # Auto trigger search
                except Exception as e:
                    st.error("Failed to load URL.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Suggestion Chips
    st.markdown("<br><p style='font-size: 0.9rem; font-weight: 600; color: #555;'>Try asking TailorTalk:</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("<div class='sugg-btn'>", unsafe_allow_html=True)
        if st.button("🔍 Find sarees similar to this", use_container_width=True): trigger_prompt = "Find sarees similar to this"
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='sugg-btn'>", unsafe_allow_html=True)
        if st.button("✨ Similar borders", use_container_width=True): trigger_prompt = "Show me designs with a similar border"
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='sugg-btn'>", unsafe_allow_html=True)
        if st.button("🎨 Similar colors", use_container_width=True): trigger_prompt = "Find sarees with similar colors"
        st.markdown("</div>", unsafe_allow_html=True)

    # Hide chat input behind an expander to match mockup cleaner aesthetic while still keeping the chat requirement
    with st.expander("Or type your own custom question..."):
        custom_prompt = st.text_input("Ask TailorTalk:", placeholder="E.g. Find sarees in blue color")
        if st.button("Ask Agent"):
            trigger_prompt = custom_prompt

# Execute Search
if trigger_prompt:
    if not st.session_state.query_image_path:
        st.error("Please upload an image or provide a URL first.")
    else:
        with st.spinner("Curating your selection..."):
            try:
                full_prompt = f"{trigger_prompt}\nImage Source: {st.session_state.query_image_path}"
                response = agent.invoke({"input": full_prompt})
                st.session_state.raw_results = searcher.search(st.session_state.query_image_path, top_k=3)
            except Exception as e:
                st.error(f"Error: {e}")

# Right Column - Featured Image & Results
with col_right:
    if not st.session_state.raw_results:
        # Match mockup perfectly: Image in a restricted container, text below
        st.markdown("""
        <div class='right-image-container'>
            <img src='https://byrappasilk.in/storage/products/featured/featured_6a3534093e08f.webp' alt='Featured Saree'>
        </div>
        <h2 style='margin-top: 1.5rem; font-size: 1.5rem;'>Curated Silks</h2>
        <p style='color: #666; font-size: 0.95rem; line-height: 1.5;'>Discover traditional craftsmanship interpreted for the modern editorial aesthetic.</p>
        """, unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='font-size: 1.5rem;'>Search Results</h2>", unsafe_allow_html=True)
        
        for idx, r in enumerate(st.session_state.raw_results):
            img_src = r['metadata'].get('image_url', '') if not os.path.exists(r['image_path']) else r['image_path']
            # Fallback if local path exists we need a base64 or let streamlit handle it.
            # Streamlit st.image handles local paths safely. Let's use st.image but styled.
            with st.container():
                st.image(img_src, use_container_width=True)
                st.markdown(f"**{r['metadata'].get('Name', 'Saree')}**")
                st.markdown(f"Similarity Score: **{r['similarity_score']:.2f}**")
                st.markdown(f"[View Details]({r['metadata'].get('Website Link', '')})")
                st.divider()
