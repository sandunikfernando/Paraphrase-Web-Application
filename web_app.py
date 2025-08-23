from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from nltk.tokenize import sent_tokenize
import streamlit as st
import time
import nltk
import os

st.set_page_config(
    page_title="NeuralText Pro",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# NLTK Setup with Error Handling
@st.cache_resource
def initialize_nltk():
    """Initialize NLTK with proper error handling for production deployment."""
    try:
        nltk_data_dir = "/tmp/nltk_data"
        os.makedirs(nltk_data_dir, exist_ok=True)
        nltk.data.path.append(nltk_data_dir)
        
        nltk.download("punkt_tab", download_dir=nltk_data_dir, quiet=True)
        nltk.download("punkt", download_dir=nltk_data_dir, quiet=True)
        return True
    except Exception as e:
        st.error(f"NLTK initialization failed: {e}")
        return False

# Model Loading with Caching and Error Handling
@st.cache_resource
def load_transformer_model():
    """Load and cache the T5-based paraphrasing model with GPU optimization."""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        with st.spinner("🤖 Loading Neural Network... Please wait."):
            tokenizer = AutoTokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
            model = AutoModelForSeq2SeqLM.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base")
            
            if device == "cuda":
                model = model.half()  # Use FP16 for GPU optimization
            
            model = model.to(device)
            model.eval()  # Set to evaluation mode
            
        return tokenizer, model, device
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None, None, None

# Initialize components
nltk_ready = initialize_nltk()
tokenizer, model, device = load_transformer_model()

# Session state initialization
if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = {}

def neural_paraphrase_sentence(
    sentence: str,
    temperature: float = 0.7,
    max_length: int = 128,
    num_beams: int = 3,
    repetition_penalty: float = 1.2
) -> str:
    """
    Advanced neural paraphrasing with optimized parameters for production use.
    
    Args:
        sentence: Input sentence to paraphrase
        temperature: Controls randomness in generation
        max_length: Maximum output length
        num_beams: Beam search width
        repetition_penalty: Penalty for repeated tokens
    
    Returns:
        Paraphrased sentence string
    """
    if not sentence.strip():
        return sentence
        
    try:
        # Tokenize input with optimization
        input_ids = tokenizer(
            f'paraphrase: {sentence}',
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=True,
        ).input_ids.to(device)
        
        # Generate with memory optimization
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                num_beams=num_beams,
                max_length=max_length,
                early_stopping=True,
                do_sample=True
            )
        
        # Decode results
        result = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return result[0] if result else sentence
        
    except Exception as e:
        st.error(f"Neural processing error: {e}")
        return sentence

def process_document(text: str) -> tuple:
    """
    Professional document processing with progress tracking and analytics.
    
    Args:
        text: Input document text
        
    Returns:
        Tuple of (paraphrased_text, processing_stats)
    """
    if not text.strip():
        return "", {}
    
    # Tokenize document into sentences
    sentences = sent_tokenize(text)
    processed_sentences = []
    
    # Initialize progress tracking
    progress_bar = st.progress(0, text="Initializing neural processing...")
    status_container = st.empty()
    
    start_time = time.time()
    
    try:
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Update progress
                progress_pct = (i + 1) / len(sentences)
                progress_bar.progress(progress_pct, text=f"Processing sentence {i+1}/{len(sentences)}")
                
                # Process sentence
                paraphrased = neural_paraphrase_sentence(sentence)
                processed_sentences.append(paraphrased)
            else:
                processed_sentences.append(sentence)
    
    except Exception as e:
        st.error(f"Document processing failed: {e}")
        return text, {}
    
    finally:
        progress_bar.empty()
        status_container.empty()
    
    # Calculate processing statistics
    processing_time = time.time() - start_time
    final_text = " ".join(processed_sentences)
    
    stats = {
        "input_words": len(text.split()),
        "output_words": len(final_text.split()),
    }
    
    return final_text, stats

# Professional Theme System
def apply_professional_theme():
        
    theme_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header */
    .neural-header {
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 20px;
        padding: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    .neural-header h1 {
        background: linear-gradient(135deg, #00d4ff, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.025em;
    }
    
    .neural-header p {
        color: rgba(255,255,255,0.8);
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Text Areas */
    .stTextArea > div > div > textarea {
        background: rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(10px) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 15px !important;
        font-family: 'Monaco', 'Menlo', monospace !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border: 2px solid #00d4ff !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff, #a855f7) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.8rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.6) !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em !important;
    }
    
    /* Stats Display */
    .neural-stats {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: rgba(255,255,255,0.9);
    }
    </style>
    """
    
    st.markdown(theme_css, unsafe_allow_html=True)

# Apply professional theme
apply_professional_theme()

# Main Header
st.markdown("""
<div class="neural-header">
    <h1>NeuralText Pro</h1>
    <p>Advanced AI-Powered Text Paraphrasing Engine</p>
</div>
""", unsafe_allow_html=True)

# Professional Layout
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### Orginal Text")
    
    input_text = st.text_area(
        "",
        height=450,
        placeholder="Enter your text here for AI-powered paraphrasing...",
        key="neural_input"
    )
    # Real-time document analytics
    if input_text:
        words = len(input_text.split())
        chars = len(input_text)
        sentences = len(sent_tokenize(input_text))
        
        st.markdown(f"""
        <div class="neural-stats">
            <p><strong>Words:</strong> {words} | <strong>Characters:</strong> {chars} | <strong>Sentences:</strong> {sentences}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Processing Button under the text area
    if st.button("Paraphrase", use_container_width=False, type="primary"):
        if input_text.strip():
            if not nltk_ready or tokenizer is None:
                st.error("Please refresh the page.")
            else:
                try:
                    # Process document with professional analytics
                    processed_text, stats = process_document(input_text)
                    
                    # Store results in session state
                    st.session_state.processed_result = processed_text
                    st.session_state.processing_stats = stats
                    
                    # Success notification
                    st.success("Paraphrasing completed successfully!")
                    
                except Exception as e:
                    st.error(f"Paraphrasing failed: {str(e)}")
        else:
            st.warning("Please provide input text for Paraphrasing.")

with col2:
    st.markdown("### Paraphrased Text")
    
    # Display Results
    if hasattr(st.session_state, 'processed_result'):
        st.text_area(
            "",
            value=st.session_state.processed_result,
            height=450,
            disabled=True,
            key="neural_output"
        )
        
        # Statistics
        if st.session_state.processing_stats:
            stats = st.session_state.processing_stats
            st.markdown(f"""
            <div class="neural-stats">
                <p><strong>Text Transformation:</strong> {stats.get('input_words', 0)} → {stats.get('output_words', 0)} words</p>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.text_area(
            "",
            value="Paraphrased text will appear here after processing...",
            height=450,
            disabled=True,
            key="neural_placeholder"
        )

# Professional Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; padding: 1.5rem;">
    <p><small><strong>NeuralText Pro</strong> | Powered by Transformer Architecture</small></p>
</div>
""", unsafe_allow_html=True)