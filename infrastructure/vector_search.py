from typing import List, Dict, Any, Optional
import json
from chromadb.types import Where, WhereDocument

from infrastructure.embeddings import get_chroma_collection
from infrastructure.logger import get_logger

logger = get_logger()

def find_nearest_neighbors(
    query: str,
    num_neighbors: int,
    where: Optional[Where] = None,
    where_document: Optional[WhereDocument] = None,
) -> List[Dict[str, Any]]:
    """
    Finds the nearest neighbors for a given query using ChromaDB, with optional filtering.

    Args:
        query: The query string.
        num_neighbors: The number of nearest neighbors to retrieve.
        where: An optional dictionary for metadata-based filtering.
        where_document: An optional dictionary for document-based filtering.

    Returns:
        A list of dictionaries, where each dictionary contains the 'id', 
        'distance', 'document', and 'metadata' of a neighbor.
    """
    logger.info(f"Finding {num_neighbors} nearest neighbors for query: '{query}' with filters: where={where}, where_document={where_document}")
    
    collection = get_chroma_collection()
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=num_neighbors,
            where=where,
            where_document=where_document,
            include=["metadatas", "documents", "distances"]
        )
        
        if not results or not results.get("ids") or not results["ids"][0]:
            logger.warning("No neighbors found or results format is invalid.")
            return []

        ids = results["ids"][0]
        distances_list = results.get("distances")
        documents_list = results.get("documents")
        metadatas_list = results.get("metadatas")

        distances = distances_list[0] if distances_list is not None else []
        documents = documents_list[0] if documents_list is not None else []
        metadatas = metadatas_list[0] if metadatas_list is not None else []
        
        neighbors = []
        for i, doc_id in enumerate(ids):
            metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
            
            # Create a mutable copy of metadata to allow modification
            if metadata:
                mutable_metadata = dict(metadata)
                if 'restricts' in mutable_metadata and isinstance(mutable_metadata['restricts'], str):
                    mutable_metadata['restricts'] = json.loads(mutable_metadata['restricts'])
                if 'numeric_restricts' in mutable_metadata and isinstance(mutable_metadata['numeric_restricts'], str):
                    mutable_metadata['numeric_restricts'] = json.loads(mutable_metadata['numeric_restricts'])
                metadata = mutable_metadata

            neighbors.append({
                "id": doc_id,
                "distance": distances[i] if distances and i < len(distances) else float('inf'),
                "document": documents[i] if documents and i < len(documents) else "",
                "metadata": metadata,
            })
        
        logger.info(f"Found {len(neighbors)} neighbors.")
        return neighbors

    except Exception as e:
        logger.error(f"ChromaDB neighbor search failed: {e}")
        return []


def upsert_datapoints(datapoints: list[dict]):
    """Upserts datapoints to the ChromaDB collection."""
    if not datapoints:
        logger.info("No datapoints to upsert.")
        return

    collection = get_chroma_collection()
    
    ids = [str(dp['datapoint_id']) for dp in datapoints]
    embeddings = [dp['embedding'] for dp in datapoints]
    
    # ChromaDB metadatas must be Dict[str, str | int | float | bool]
    # Our 'restricts' are lists of dicts, which are not supported.
    # We will serialize them to JSON strings.
    metadatas = []
    for dp in datapoints:
        metadata = {}
        if 'restricts' in dp:
            # Assuming 'restricts' is a list of dictionaries
            metadata['restricts'] = json.dumps(dp['restricts'])
        if 'numeric_restricts' in dp:
            metadata['numeric_restricts'] = json.dumps(dp['numeric_restricts'])
        metadatas.append(metadata)

    try:
        logger.info(f"Upserting {len(datapoints)} datapoints to ChromaDB collection.")
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        logger.info(f"Successfully upserted {len(datapoints)} datapoints.")
    except Exception as e:
        logger.error(f"Failed to upsert to ChromaDB: {e}")
        raise Exception("Failed to upsert to ChromaDB")
