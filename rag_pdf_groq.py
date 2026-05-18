import os
from pathlib import Path
import requests
import xml.etree.ElementTree as ET

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter



DATA_DIR = Path("data")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GROQ_MODEL = "llama-3.3-70b-versatile"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 3




def load_pdf_documents(data_dir: Path):

    pdf_paths = sorted(data_dir.rglob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            "Aucun PDF trouve dans le dossier 'data'."
        )

    documents = []

    for pdf_path in pdf_paths:

        loader = PyPDFLoader(str(pdf_path))

        pages = loader.load()

        for page in pages:
            page.metadata["source"] = pdf_path.name

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

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return FAISS.from_documents(chunks, embeddings)



def build_prompt():

    template = """
Tu es un assistant academique intelligent specialise en IA et informatique.

Tu aides les etudiants a comprendre :
- les cours ;
- les concepts scientifiques ;
- les articles de recherche.

Tu peux utiliser :
- les supports PDF ;
- les articles scientifiques provenant de arXiv.

Consignes importantes :
- Reponds en francais.
- Explique simplement.
- Donne des exemples si possible.
- Reponds de facon pedagogique.
- Utilise uniquement le contexte fourni.
- Si l'information est absente, dis-le clairement.
- Termine par une ligne "Sources :".

Contexte :
{context}

Question :
{question}
"""

    return PromptTemplate.from_template(template)




def format_context(docs):

    parts = []

    for i, doc in enumerate(docs, start=1):

        source = doc.metadata.get(
            "source",
            "source_inconnue"
        )

        page = doc.metadata.get("page", "?")

        if isinstance(page, int):
            page = page + 1

        parts.append(
            f"[Extrait {i} | source={source} | page={page}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(parts)




def format_sources(docs):

    unique_sources = []

    for doc in docs:

        source = doc.metadata.get(
            "source",
            "source_inconnue"
        )

        page = doc.metadata.get("page", "?")

        if isinstance(page, int):
            page = page + 1

        item = f"{source} (page {page})"

        if item not in unique_sources:
            unique_sources.append(item)

    return ", ".join(unique_sources)




def search_arxiv(query, max_results=3):

    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query=all:{query}"
        f"&start=0"
        f"&max_results={max_results}"
    )

    response = requests.get(url)

    if response.status_code != 200:
        return []

    root = ET.fromstring(response.text)

    articles = []

    for entry in root.findall(
        "{http://www.w3.org/2005/Atom}entry"
    ):

        title = entry.find(
            "{http://www.w3.org/2005/Atom}title"
        ).text

        summary = entry.find(
            "{http://www.w3.org/2005/Atom}summary"
        ).text

        articles.append({
            "title": title.strip(),
            "summary": summary.strip()
        })

    return articles




def format_arxiv_context(articles):

    parts = []

    for i, article in enumerate(articles, start=1):

        parts.append(
            f"[Article {i}]\n"
            f"Titre : {article['title']}\n"
            f"Resume : {article['summary']}"
        )

    return "\n\n".join(parts)




def answer_academic_question(
    question,
    retriever,
    llm,
    prompt
):

    docs = retriever.invoke(question)

    pdf_context = format_context(docs)

    arxiv_articles = search_arxiv(question)

    arxiv_context = format_arxiv_context(
        arxiv_articles
    )

  
    full_context = (
        "=== SUPPORTS DE COURS ===\n\n"
        + pdf_context
        + "\n\n"
        + "=== ARTICLES SCIENTIFIQUES arXiv ===\n\n"
        + arxiv_context
    )

    final_prompt = prompt.format(
        context=full_context,
        question=question
    )

    response = llm.invoke(final_prompt).content

    sources = format_sources(docs)

    if "Sources :" not in response:
        response += f"\n\nSources : {sources}"

    return response, docs




def main():

    load_dotenv()

    groq_api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not groq_api_key:
        raise EnvironmentError(
            "La variable GROQ_API_KEY est absente."
        )

    print("Chargement des PDF...")

    documents = load_pdf_documents(DATA_DIR)

    print(
        f"Nombre total de pages : {len(documents)}"
    )

    print("Decoupage en chunks...")

    chunks = split_documents(documents)

    print(
        f"Nombre total de chunks : {len(chunks)}"
    )

    print("Creation de FAISS...")

    vectorstore = create_vectorstore(chunks)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )

    llm = ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        api_key=groq_api_key,
    )

    prompt = build_prompt()

    print("\nAssistant Academique pret.")
    print("Tapez votre question.")
    print("Tapez 'quit' pour quitter.\n")

    while True:

        question = input("Question > ").strip()

        if not question:
            print("Veuillez saisir une question.\n")
            continue

        if question.lower() in {
            "quit",
            "exit",
            "q"
        }:
            print("Fin du programme.")
            break

        answer, docs = answer_academic_question(
            question,
            retriever,
            llm,
            prompt,
        )

        print("\n===== REPONSE =====\n")

        print(answer)

        print("\n===== CHUNKS RECUPERES =====\n")

        for i, doc in enumerate(docs, start=1):

            source = doc.metadata.get(
                "source",
                "source_inconnue"
            )

            page = doc.metadata.get("page", "?")

            if isinstance(page, int):
                page = page + 1

            preview = doc.page_content[:250]

            preview = preview.replace(
                "\n",
                " "
            )

            print(f"{i}. {source} | page {page}")

            print(f"   {preview}...\n")




if __name__ == "__main__":
    main()