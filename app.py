import os
import json
from pathlib import Path

import requests
import xml.etree.ElementTree as ET

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─── Configuration ────────────────────────────────────────────────────────────

DATA_DIR        = Path("data")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL      = "llama-3.3-70b-versatile"
CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 150
TOP_K           = 3

load_dotenv()

app = Flask(__name__)

# Global singletons (built once at startup)
_retriever = None
_llm       = None
_prompt    = None
_subjects  = []


# ─── Startup : charger les PDF et construire FAISS ───────────────────────────

def load_pdf_documents(data_dir: Path):
    pdf_paths = sorted(data_dir.rglob("*.pdf"))
    documents = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        pages  = loader.load()
        for page in pages:
            page.metadata["source"]  = pdf_path.name
            page.metadata["subject"] = pdf_path.parent.name
        documents.extend(pages)
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(chunks, embeddings)


def get_subjects():
    if not DATA_DIR.exists():
        return []
    return [d.name for d in sorted(DATA_DIR.iterdir()) if d.is_dir()]


def init_rag():
    global _retriever, _llm, _prompt, _subjects
    print("[INIT] Chargement des PDF...")
    docs   = load_pdf_documents(DATA_DIR)
    chunks = split_documents(docs)
    vs     = create_vectorstore(chunks)
    _retriever = vs.as_retriever(search_kwargs={"k": TOP_K})

    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise EnvironmentError("GROQ_API_KEY manquante dans .env")

    _llm = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=groq_key)

    _prompt = PromptTemplate.from_template("""
Tu es un assistant académique intelligent spécialisé en IA et informatique.
Tu aides les étudiants à comprendre les cours, concepts scientifiques et articles.

Consignes :
- Réponds en français.
- Explique simplement avec des exemples si possible.
- Utilise uniquement le contexte fourni.
- Si l'information est absente, dis-le clairement.
- Termine par une ligne "Sources :".

Contexte :
{context}

Question :
{question}
""")

    _subjects = get_subjects()
    print(f"[INIT] {len(docs)} pages | {len(chunks)} chunks | {len(_subjects)} matières")


# ─── Helpers RAG ─────────────────────────────────────────────────────────────

def format_context(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        source  = doc.metadata.get("source", "?")
        subject = doc.metadata.get("subject", "?")
        page    = doc.metadata.get("page", "?")
        if isinstance(page, int):
            page += 1
        parts.append(
            f"[Extrait {i} | matière={subject} | source={source} | page={page}]\n"
            f"{doc.page_content}"
        )
    return "\n\n".join(parts)


def format_sources(docs):
    seen, items = set(), []
    for doc in docs:
        source  = doc.metadata.get("source", "?")
        subject = doc.metadata.get("subject", "?")
        page    = doc.metadata.get("page", "?")
        if isinstance(page, int):
            page += 1
        key = f"{subject}/{source} (p.{page})"
        if key not in seen:
            seen.add(key)
            items.append(key)
    return ", ".join(items)


# ─── Helpers ArXiv ───────────────────────────────────────────────────────────

def search_arxiv(query: str, max_results: int = 5):
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query=all:{requests.utils.quote(query)}"
        f"&start=0&max_results={max_results}"
    )
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        root     = ET.fromstring(resp.text)
        ns       = "{http://www.w3.org/2005/Atom}"
        articles = []
        for entry in root.findall(f"{ns}entry"):
            title        = entry.find(f"{ns}title").text or ""
            summary      = entry.find(f"{ns}summary").text or ""
            link_el      = entry.find(f"{ns}id")
            link         = link_el.text.strip() if link_el is not None else ""
            authors      = [
                a.find(f"{ns}name").text
                for a in entry.findall(f"{ns}author")
                if a.find(f"{ns}name") is not None
            ]
            pub_el       = entry.find(f"{ns}published")
            published    = pub_el.text[:10] if pub_el is not None else ""
            articles.append({
                "title":     title.strip(),
                "summary":   summary.strip(),
                "link":      link,
                "authors":   authors[:3],
                "published": published,
            })
        return articles
    except Exception as e:
        print(f"[ArXiv] Erreur : {e}")
        return []


# ─── Outils Agent (TP 3) ─────────────────────────────────────────────────────

@tool
def calculer_moyenne(notes_json: str) -> str:
    """Calcule la moyenne, le min et le max à partir d'une liste JSON de notes."""
    try:
        notes = json.loads(notes_json)
        if not isinstance(notes, list) or not notes:
            return "Erreur : liste JSON non vide requise, ex: [12, 15, 9]."
        notes   = [float(n) for n in notes]
        moyenne = sum(notes) / len(notes)
        return (
            f"Moyenne : {moyenne:.2f}\n"
            f"Minimum : {min(notes):.2f}\n"
            f"Maximum : {max(notes):.2f}"
        )
    except Exception as exc:
        return f"Erreur : {exc}"


@tool
def generer_plan_revision(sujet: str) -> str:
    """Génère un plan de révision structuré en 5 étapes pour un sujet donné."""
    sujet = sujet.strip()
    if not sujet:
        return "Erreur : sujet requis."
    return (
        f"Plan de révision — {sujet}\n"
        "1. Relire les définitions et notions de base.\n"
        "2. Identifier 5 mots-clés essentiels.\n"
        "3. Refaire un exemple concret ou mini exercice.\n"
        "4. Résumer en 5 lignes maximum.\n"
        "5. Se tester avec 3 questions de révision."
    )


@tool
def donner_definition_simple(mot: str) -> str:
    """Donne une définition simple et pédagogique d'un concept ou mot clé."""
    mot = mot.strip()
    if not mot:
        return "Erreur : mot requis."
    return f"Le modèle va définir simplement : {mot}"


AGENT_TOOLS         = [calculer_moyenne, generer_plan_revision, donner_definition_simple]
AGENT_TOOLS_BY_NAME = {t.name: t for t in AGENT_TOOLS}


def run_agent(question: str) -> str:
    groq_key = os.getenv("GROQ_API_KEY")
    llm      = ChatGroq(model=GROQ_MODEL, temperature=0, api_key=groq_key)
    llm_with = llm.bind_tools(AGENT_TOOLS)

    messages = [
        SystemMessage(content=(
            "Tu es un assistant pédagogique pour étudiant en IA. "
            "Tu peux calculer des moyennes, générer des plans de révision "
            "et donner des définitions simples. "
            "Utilise un outil uniquement si c'est pertinent. "
            "Réponds toujours en français de façon claire et concise."
        )),
        HumanMessage(content=question),
    ]

    first = llm_with.invoke(messages)
    messages.append(first)

    if first.tool_calls:
        for tc in first.tool_calls:
            selected = AGENT_TOOLS_BY_NAME.get(tc["name"])
            if selected:
                output = selected.invoke(tc["args"])
                messages.append(ToolMessage(content=output, tool_call_id=tc["id"]))
        final = llm_with.invoke(messages)
        return final.content

    return first.content


# ─── Routes Flask ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", subjects=_subjects)


@app.route("/api/subjects")
def api_subjects():
    return jsonify({"subjects": _subjects})


# 1. RAG — Questions sur les PDF
@app.route("/api/rag", methods=["POST"])
def api_rag():
    data     = request.get_json()
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question vide."}), 400

    docs        = _retriever.invoke(question)
    pdf_context = format_context(docs)
    sources     = format_sources(docs)

    final_prompt = _prompt.format(context=pdf_context, question=question)
    response     = _llm.invoke(final_prompt).content

    if "Sources :" not in response:
        response += f"\n\n**Sources :** {sources}"

    chunks_info = []
    for doc in docs:
        page = doc.metadata.get("page", "?")
        if isinstance(page, int):
            page += 1
        chunks_info.append({
            "subject": doc.metadata.get("subject", "?"),
            "source":  doc.metadata.get("source", "?"),
            "page":    page,
            "preview": doc.page_content[:200].replace("\n", " "),
        })

    return jsonify({"answer": response, "chunks": chunks_info})


# 2. ArXiv — Chercher des articles
@app.route("/api/arxiv", methods=["POST"])
def api_arxiv():
    data    = request.get_json()
    query   = (data.get("query") or "").strip()
    max_res = int(data.get("max_results", 5))
    if not query:
        return jsonify({"error": "Requête vide."}), 400

    articles = search_arxiv(query, max_res)
    return jsonify({"articles": articles})


# 3. Résumer un cours / chapitre
@app.route("/api/resume", methods=["POST"])
def api_resume():
    data  = request.get_json()
    sujet = (data.get("sujet") or "").strip()
    if not sujet:
        return jsonify({"error": "Sujet vide."}), 400

    docs        = _retriever.invoke(sujet)
    pdf_context = format_context(docs)
    sources     = format_sources(docs)

    prompt = f"""
Tu es un assistant académique. À partir des extraits de cours fournis,
résume le sujet suivant de façon claire, structurée et pédagogique en français.
Utilise des titres et des points clés. Termine par "Sources :".

Sujet : {sujet}

Extraits :
{pdf_context}
"""
    response = _llm.invoke(prompt).content
    if "Sources :" not in response:
        response += f"\n\n**Sources :** {sources}"

    return jsonify({"answer": response})


# 4. Expliquer un concept simplement
@app.route("/api/expliquer", methods=["POST"])
def api_expliquer():
    data    = request.get_json()
    concept = (data.get("concept") or "").strip()
    if not concept:
        return jsonify({"error": "Concept vide."}), 400

    docs        = _retriever.invoke(concept)
    pdf_context = format_context(docs)
    sources     = format_sources(docs)

    prompt = f"""
Tu es un enseignant expert. Explique le concept suivant de façon très simple,
comme si tu expliquais à un lycéen. Utilise une analogie concrète et un exemple
du quotidien. Réponds en français. Termine par "Sources :".

Concept : {concept}

Contexte :
{pdf_context}
"""
    response = _llm.invoke(prompt).content
    if "Sources :" not in response:
        response += f"\n\n**Sources :** {sources}"

    return jsonify({"answer": response})


# 5. Générer un plan de révision (agent)
@app.route("/api/plan", methods=["POST"])
def api_plan():
    data  = request.get_json()
    sujet = (data.get("sujet") or "").strip()
    if not sujet:
        return jsonify({"error": "Sujet vide."}), 400

    # Utiliser directement le LLM au lieu de l'agent
    prompt = f"""
Tu es un assistant pédagogique. Génère un plan de révision détaillé et structuré 
en français pour le sujet suivant. Utilise des titres, sous-titres et points clés.

Sujet : {sujet}
"""
    try:
        response = _llm.invoke(prompt).content
        if not response:
            response = "Impossible de générer un plan pour ce sujet."
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# 6. Agent général (calcul, définition, etc.)
@app.route("/api/agent", methods=["POST"])
def api_agent():
    data     = request.get_json()
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question vide."}), 400

    answer = run_agent(question)
    return jsonify({"answer": answer})


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_rag()
    app.run(debug=True, port=5000)