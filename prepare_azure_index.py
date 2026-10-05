"""Prepare the Chroma index that is packaged into the Azure container image."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from article_index import article_documents, read_articles


ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
PDF_PATH = Path(os.getenv("CODEX_PDF_PATH", ROOT / "data" / "Codex_over_het_welzijn_op_het_werk.pdf"))
SEED_ROOT = ROOT / "deployment" / "chroma_seed"
COLLECTION_NAME = "codex_welzijn_artikels"


def pdf_signature(pdf_path: Path) -> str:
    digest = hashlib.sha256()
    with pdf_path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    digest.update(b"article-index-v5-passage-search-highlight")
    return digest.hexdigest()[:20]


def main() -> None:
    if os.getenv("AI_PROVIDER", "").strip().lower() != "azure":
        raise SystemExit("Stel in .env eerst AI_PROVIDER=azure in.")

    api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "")
    if not all((api_key, endpoint, deployment)):
        raise SystemExit(
            "Vul AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT en "
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT in .env in."
        )

    if not PDF_PATH.exists():
        raise SystemExit(f"Codex-PDF niet gevonden: {PDF_PATH}")

    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", deployment)
    index_path = SEED_ROOT / f"{pdf_signature(PDF_PATH)}-azure-{safe_model}"
    marker = index_path / ".index_complete"
    if index_path.exists():
        if marker.exists() and (index_path / "chroma.sqlite3").exists():
            print(f"De index bestaat al: {index_path.name}")
            return
        raise SystemExit(
            f"Er staat al een onvolledige indexmap: {index_path}. "
            "Bewaar of hernoem die map eerst; ze wordt niet automatisch overschreven."
        )

    articles = read_articles(PDF_PATH)
    documents = article_documents(articles)
    if not documents:
        raise SystemExit("Er zijn geen artikelen gevonden om te indexeren.")

    SEED_ROOT.mkdir(parents=True, exist_ok=True)
    # Use the same OpenAI-compatible v1 endpoint as the running app.
    # With v1, the deployment name is passed as the model and no dated API
    # version is needed.
    base_url = endpoint if endpoint.endswith("/openai/v1") else endpoint + "/openai/v1"
    embeddings = OpenAIEmbeddings(
        model=deployment,
        api_key=api_key,
        base_url=base_url + "/",
    )
    ids = [
        f"artikel-{document.metadata['article_number']}-onderdeel-{document.metadata['part_index']}"
        for document in documents
    ]
    print(f"Bezig met embeddings maken voor {len(documents):,} artikelonderdelen…")
    try:
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            ids=ids,
            collection_name=COLLECTION_NAME,
            persist_directory=str(index_path),
        )
        del vectorstore
        marker.write_text(str(len(documents)), encoding="utf-8")
    except Exception:
        shutil.rmtree(index_path, ignore_errors=True)
        raise
    print(f"Klaar. De index wordt bij de volgende Docker-build in de container opgenomen.")


if __name__ == "__main__":
    main()
