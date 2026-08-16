"""
Model Listing & Embedding Routes
"""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from src.config import MODEL_NAME
from src.server.schemas import EmbeddingRequest
from src.server.auth import get_api_key

router = APIRouter(prefix="/v1", tags=["Models"])

@router.get("/models")
async def list_models(kd: Dict[str, Any] = Depends(get_api_key)):
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "ozaa77",
                "permission": [],
                "root": MODEL_NAME,
                "parent": None,
            }
        ]
    }

@router.post("/embeddings")
async def create_embeddings(body: EmbeddingRequest, kd: Dict[str, Any] = Depends(get_api_key)):
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Embeddings are not supported for this generative CausalLM model.",
    )
