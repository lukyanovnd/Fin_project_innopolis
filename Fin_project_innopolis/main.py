import streamlit as st
import os
import tempfile
import numpy as np
import re
import torch
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import pymorphy3
import pdfplumber
from transformers import AutoTokenizer, AutoModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from datetime import datetime
from typing import List, Dict

# ================= КОНФИГУРАЦИЯ =================
CORPUS_DIR = "/var/lib/corpus_pdfs"
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "plagiarism_corpus"
BERT_MODEL_NAME = "DeepPavlov/rubert-base-cased"
MIN_SENTENCE_LENGTH = 5
MAX_TOKEN_LENGTH = 128
UPSERT_BATCH_SIZE = 500

# ================= ИНИЦИАЛИЗАЦИЯ =================
@st.cache_resource(show_spinner=False)
def setup_environment():
    try:
        os.makedirs(CORPUS_DIR, exist_ok=True)
    except PermissionError:
        globals()["CORPUS_DIR"] = "./local_corpus_pdfs"
        os.makedirs(globals()["CORPUS_DIR"], exist_ok=True)
        st.warning("⚠️ Нет прав на /var/lib. Данные сохраняются в ./local_corpus_pdfs")
    return globals()["CORPUS_DIR"]

setup_environment()

@st.cache_resource(show_spinner=False)
def init_nltk():
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    return True

init_nltk()

@st.cache_resource(show_spinner="Подключение к Qdrant...")
def get_qdrant_client():
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()  # Проверка соединения
        return client
    except Exception as e:
        st.error(f"❌ Не удалось подключиться к Qdrant ({QDRANT_HOST}:{QDRANT_PORT}): {e}")
        return None

# ================= ТЕКСТОВАЯ ОБРАБОТКА =================
def extract_sentences_from_pdf(file_path: str) -> List[str]:
    if not os.path.exists(file_path):
        return []
    full_text = []
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text: full_text.append(text)
    except Exception as e:
        st.error(f"Ошибка чтения PDF: {e}")
        return []

    combined = " ".join(full_text)
    try:
        sentences = sent_tokenize(combined, language="russian")
    except Exception:
        sentences = combined.split(".")

    russian_stopwords = set(stopwords.words("russian"))
    morph = pymorphy3.MorphAnalyzer()
    cleaned = []
    for s in sentences:
        if len(s) <= MIN_SENTENCE_LENGTH:
            continue
        s = s.replace("\n", " ").replace("- ", "").lower()
        s_clean = re.sub(r'[\d+(?:,\s*\d+)*]|[^\w\s]', '', s)
        words = s_clean.split()
        filtered = [w for w in words if w not in russian_stopwords and w.strip()]
        lemmatized = []
        for w in filtered:
            try:
                lemmatized.append(morph.parse(w)[0].normal_form)
            except Exception:
                lemmatized.append(w)
        if lemmatized:
            cleaned.append(" ".join(lemmatized))
    return cleaned

# ================= BERT МОДЕЛЬ =================
@st.cache_resource(show_spinner="Загрузка BERT модели...")
def get_bert_model():
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    model = AutoModel.from_pretrained(BERT_MODEL_NAME)
    model.eval()
    return tokenizer, model

def compute_embeddings_batch(tokenizer, model, texts: List[str]) -> List[np.ndarray]:
    embeddings = []
    for text in texts:
        try:
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKEN_LENGTH)
            with torch.no_grad():
                outputs = model(**inputs)
            emb = torch.mean(outputs.last_hidden_state[0], dim=0).numpy()
            embeddings.append(emb)
        except Exception:
            embeddings.append(np.zeros(768))
    return embeddings

# ================= QDRANT ИНТЕГРАЦИЯ (ИСПРАВЛЕНО) =================
def prepare_qdrant_collection(client):
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        client.delete_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )

def upsert_to_qdrant(client, file_id: int, file_name: str, sentences: List[str], embeddings: List[np.ndarray]):
    points = []
    for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
        points.append(PointStruct(
            id=file_id * 1_000_000 + i,
            vector=emb.tolist(),
            payload={
                "file_id": file_id,
                "file_name": file_name,
                "sentence_index": i,
                "text": sent
            }
        ))
    
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[i:i + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

def search_in_qdrant(client, query_embeddings: List[np.ndarray]) -> List[Dict]:
    results = []
    for emb in query_embeddings:
        # ✅ ИСПОЛЬЗУЕМ АКТУАЛЬНЫЙ API (qdrant-client >= 1.7)
        search_res = client.query_points(
            collection_name=COLLECTION_NAME,
            query=emb.tolist(),
            limit=1
        )
        if search_res.points:
            hit = search_res.points[0]
            results.append({
                "score": hit.score,
                "file_name": hit.payload.get("file_name", "Unknown"),
                "sentence_index": hit.payload.get("sentence_index", -1)
            })
        else:
            results.append({"score": 0.0, "file_name": "Нет данных", "sentence_index": -1})
    return results

# ================= ОТЧЕТ =================
def generate_report(similarity_results: List[Dict], check_file_name: str) -> str:
    lines = [
        "=" * 80, "ОТЧЕТ О ПРОВЕРКЕ НА ПЛАГИАТ", "=" * 80,
        f"Дата проверки: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}",
        f"Проверяемый файл: {check_file_name}",
        f"Всего предложений проверено: {len(similarity_results)}", "",
        "РЕЗУЛЬТАТЫ:", "-" * 80
    ]
    
    if not similarity_results:
        lines.append("Нет данных для отображения")
        return "\n".join(lines)

    high_similarity_count = 0
    threshold = 0.8
    for i, res in enumerate(similarity_results):
        percent = float(res["score"])
        percent_str = f"{percent:.2f}".replace(".", ",")
        marker = "ВЫСОКАЯ СХОЖЕСТЬ" if percent >= threshold else ""
        if percent >= threshold:
            high_similarity_count += 1
            
        lines.append(
            f"Предложение №{i+1}: Схожесть с предложением №{res['sentence_index']+1} "
            f"из файла '{res['file_name']}' = {percent_str} {marker}"
        )

    lines.extend(["", "-" * 80, "СТАТИСТИКА:",
                  f"Всего предложений: {len(similarity_results)}",
                  f"Высокая схожесть ( >{threshold*100}%): {high_similarity_count}",
                  f"Процент высокой схожести: {(high_similarity_count/len(similarity_results)*100):.2f}%" if similarity_results else "0%",
                  "=" * 80])
    return "\n".join(lines)

# ================= STREAMLIT UI =================
st.set_page_config(page_title="Проверка на плагиат (Qdrant)", layout="wide")
st.title("🕵️‍♂️ Система проверки на плагиат | Streamlit + Qdrant")

client = get_qdrant_client()
if not client:
    st.stop()

tab1, tab2, tab3 = st.tabs(["📚 Загрузка корпуса", "🧮 Расчет эмбеддингов", "🔍 Проверка текста"])

with tab1:
    st.header("Загрузка PDF-файлов корпуса")
    uploaded_corpus = st.file_uploader("Выберите один или несколько PDF файлов", type=["pdf"], accept_multiple_files=True)
    if uploaded_corpus:
        if st.button("💾 Сохранить файлы корпуса", type="primary"):
            saved = 0
            for f in uploaded_corpus:
                save_path = os.path.join(CORPUS_DIR, f.name)
                with open(save_path, "wb") as out_f:
                    out_f.write(f.getvalue())
                saved += 1
            st.success(f"✅ Успешно сохранено файлов: {saved}")
    st.divider()
    st.info(f"📂 Текущие файлы в корпусе (`{CORPUS_DIR}`):")
    try:
        files = [f for f in os.listdir(CORPUS_DIR) if f.endswith(".pdf")]
        if files:
            for f in files: st.code(f"- {f}", language=None)
        else: st.caption("Папка пуста.")
    except Exception as e: st.error(f"Ошибка чтения папки: {e}")

with tab2:
    st.header("Расчет и сохранение векторных представлений в Qdrant")
    if st.button("🚀 Рассчитать эмбеддинги и записать в Qdrant", type="primary"):
        files = sorted([f for f in os.listdir(CORPUS_DIR) if f.endswith(".pdf")])
        if not files:
            st.warning("⚠️ Папка корпуса пуста. Загрузите файлы сначала.")
        else:
            with st.spinner("Инициализация модели и коллекции..."):
                tokenizer, model = get_bert_model()
                prepare_qdrant_collection(client)
            progress_bar = st.progress(0)
            status_text = st.empty()
            for i, f_name in enumerate(files):
                f_path = os.path.join(CORPUS_DIR, f_name)
                status_text.text(f"Обработка: {f_name}")
                sentences = extract_sentences_from_pdf(f_path)
                if sentences:
                    embs = compute_embeddings_batch(tokenizer, model, sentences)
                    upsert_to_qdrant(client, i, f_name, sentences, embs)
                progress_bar.progress((i + 1) / len(files))
            status_text.text("✅ Готово!")
            st.success(f"Эмбеддинги для {len(files)} файлов успешно загружены в коллекцию '{COLLECTION_NAME}'.")

with tab3:
    st.header("Анализ проверяемого документа")
    check_file = st.file_uploader("Загрузите PDF для проверки", type=["pdf"])
    if check_file:
        if st.button("🔍 Рассчитать схожесть и сформировать отчет", type="primary"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(check_file.getvalue())
                tmp_path = tmp.name
            sentences = extract_sentences_from_pdf(tmp_path)
            if not sentences:
                st.error("❌ Не удалось извлечь текст из файла.")
            else:
                tokenizer, model = get_bert_model()
                with st.spinner("Вычисление эмбеддингов проверяемого текста..."):
                    query_embs = compute_embeddings_batch(tokenizer, model, sentences)
                with st.spinner("Поиск наиболее похожих предложений в Qdrant..."):
                    similarity_results = search_in_qdrant(client, query_embs)
                report = generate_report(similarity_results, check_file.name)
                st.divider()
                st.subheader("📄 Отчет")
                st.text_area("Результат проверки", report, height=400)
                st.download_button(
                    label="📥 Скачать отчет (.txt)",
                    data=report.encode("utf-8"),
                    file_name=f"plagiarism_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    mime="text/plain"
                )