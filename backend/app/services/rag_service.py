import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from app.rag.vector_store import FAISSVectorStoreManager
from app.models.voice import GroundedSource
from app.core.config import settings

logger = logging.getLogger(__name__)

# Multilingual Crop Synonym Dictionary for query entity extraction
CROP_KEYWORDS = {
    "Cotton": ["cotton", "कापूस", "कपास", "कापुस", "पत्ती"],
    "Wheat": ["wheat", "गहू", "गेहूं", "गेंहू", "कनक"],
    "Soybean": ["soybean", "सोयाबीन", "सोयाबिन", "सोया"],
    "Rice": ["rice", "paddy", "धान", "भात", "तांदूळ", "चावल"],
    "Sugarcane": ["sugarcane", "ऊस", "गन्ना"],
    "Tomato": ["tomato", "टोमॅटो", "टमाटर"],
    "Onion": ["onion", "कांदा", "प्याज"],
}

class RAGService:

    @classmethod
    def extract_crop_from_query(cls, query: str) -> Optional[str]:
        """Infers the crop mentioned in the farmer's query in English, Marathi, or Hindi."""
        query_lower = query.lower()
        for crop_name, synonyms in CROP_KEYWORDS.items():
            for syn in synonyms:
                if syn in query_lower:
                    return crop_name
        return None

    @classmethod
    def retrieve_grounded_context(
        cls,
        query: str,
        crop: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Tuple[str, List[GroundedSource], bool]:
        """
        Performs semantic vector search in FAISS for the agricultural query.
        Returns:
            - context_string: Formatted string of retrieved context chunks.
            - sources: Structured list of GroundedSource objects.
            - is_grounded: True if high-relevance trusted documents were retrieved.
        """
        k = top_k or settings.TOP_K_RESULTS
        inferred_crop = crop or cls.extract_crop_from_query(query)

        filter_dict = {"crop": inferred_crop} if inferred_crop else None
        docs = FAISSVectorStoreManager.search_similar(query=query, top_k=k, filter_dict=filter_dict)

        if not docs and filter_dict:
            # Fallback to global search if crop-specific filter found nothing
            logger.info(f"No documents found with filter {filter_dict}. Falling back to global search.")
            docs = FAISSVectorStoreManager.search_similar(query=query, top_k=k)

        if not docs:
            logger.warning(f"No relevant documents retrieved for query: '{query}'")
            return (
                "NO_TRUSTED_CONTEXT_FOUND: No verified ICAR/KVK advisory matches this specific query in the local agricultural database.",
                [],
                False
            )

        context_chunks = []
        sources = []
        best_score = float("inf")

        for idx, doc in enumerate(docs):
            source_file = doc.metadata.get("source_file", "ICAR/KVK Advisory Guide")
            category = doc.metadata.get("category", "General Agronomy")
            source_org = doc.metadata.get("source", "Agricultural Research Institute")
            doc_crop = doc.metadata.get("crop", "General")
            doc_state = doc.metadata.get("state", "India")
            content = doc.page_content.strip()
            score = doc.metadata.get("score", 1.0)
            if score < best_score:
                best_score = score

            context_chunks.append(
                f"[Source {idx+1} ({source_file} | Org: {source_org} | Crop: {doc_crop} | Region: {doc_state} | Category: {category})]:\n{content}"
            )
            sources.append(
                GroundedSource(
                    title=f"{source_file} ({source_org})",
                    category=category,
                    excerpt=content[:220] + "..." if len(content) > 220 else content,
                    relevance_score=round(score, 4) if score is not None else None
                )
            )

        # In L2 distance / FAISS, smaller distance = more similar.
        # Typically distance < 1.2 indicates strong semantic match with sentence-transformers.
        is_grounded = len(docs) > 0 and (best_score <= 1.25 or best_score == 0.0)

        context_string = "\n\n".join(context_chunks)
        return context_string, sources, is_grounded
