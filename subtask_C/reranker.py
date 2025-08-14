import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class RerankResult:
    """Result of reranking operation"""
    candidates: List[Dict]
    scores: List[float]
    original_indices: List[int]


class ReRanker:
    """
    re-ranker using BAAI/bge-reranker-v2-m3
    Provides semantic re-ranking for candidate matches
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = None, verbose: bool = False):
        self.model_name = model_name
        self.verbose = verbose
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        if self.verbose:
            print(f"Initializing HuggingFace re-ranker: {model_name}")
            print(f"Device: {self.device}")
        
        # Initialize model and tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            if self.verbose:
                print("Re-ranker model loaded successfully")
                
        except Exception as e:
            if self.verbose:
                print(f"Failed to load re-ranker model: {e}")
            # Don't set model to None, raise the exception instead
            raise RuntimeError(f"Failed to initialize HuggingFace re-ranker model '{model_name}': {e}")
    
    def is_available(self) -> bool:
        """Check if the re-ranker model is available"""
        return self.model is not None and self.tokenizer is not None
    
    def prepare_pairs(self, query: str, candidates: List[Dict]) -> List[Tuple[str, str]]:
        """Prepare query-candidate pairs for re-ranking"""
        pairs = []
        
        for candidate in candidates:
            # Extract text from candidate - handle both 'text' and 'Matn' fields
            candidate_text = candidate.get('text', candidate.get('Matn', ''))
            if candidate_text:
                pairs.append((query, candidate_text))
        
        return pairs
    
    def calculate_similarity_scores(self, pairs: List[Tuple[str, str]], batch_size: int = 16) -> List[float]:
        """Calculate similarity scores using the re-ranker model"""
        if not self.is_available():
            return [0.0] * len(pairs)
        
        scores = []
        
        try:
            # Process in batches to manage memory
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]
                
                # Tokenize batch
                batch_inputs = []
                for query, passage in batch_pairs:
                    # Format input as expected by the model
                    input_text = f"{query} [SEP] {passage}"
                    batch_inputs.append(input_text)
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_inputs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    batch_scores = torch.sigmoid(outputs.logits).squeeze(-1)
                    
                    # Convert to list
                    if batch_scores.dim() == 0:
                        batch_scores = [batch_scores.item()]
                    else:
                        batch_scores = batch_scores.cpu().numpy().tolist()
                    
                    scores.extend(batch_scores)
        
        except Exception as e:
            if self.verbose:
                print(f"Error in re-ranking: {e}")
            # Return default scores on error
            scores = [0.0] * len(pairs)
        
        return scores
    
    def rerank_candidates(self, query: str, candidates: List[Dict], top_k: int = 10, 
                         combine_with_original: bool = True, alpha: float = 0.7) -> RerankResult:
        """
        Re-rank candidates using bge model
        
        Args:
            query: Query text
            candidates: List of candidate matches
            top_k: Number of top results to return
            combine_with_original: Whether to combine with original similarity scores
            alpha: Weight for original scores (1-alpha for reranker scores)
        """
        if not candidates:
            return RerankResult(candidates=[], scores=[], original_indices=[])
        
        if len(candidates) == 1:
            return RerankResult(
                candidates=candidates,
                scores=[candidates[0].get('similarity', 0.0)],
                original_indices=[0]
            )
        
        # Limit candidates for performance - optimized
        candidates_to_rerank = candidates[:min(len(candidates), 30)]
        
        if self.verbose and len(candidates) > 50:
            print(f"Re-ranking top {len(candidates_to_rerank)} of {len(candidates)} candidates")
        
        # Prepare query-candidate pairs
        pairs = self.prepare_pairs(query, candidates_to_rerank)
        
        if not pairs:
            return RerankResult(candidates=[], scores=[], original_indices=[])
        
        # Get re-ranking scores
        rerank_scores = self.calculate_similarity_scores(pairs)
        
        # Combine with original scores if requested
        final_scores = []
        for i, candidate in enumerate(candidates_to_rerank):
            original_score = candidate.get('similarity', 0.0)
            rerank_score = rerank_scores[i] if i < len(rerank_scores) else 0.0
            
            if combine_with_original and self.is_available():
                # Weighted combination of original and reranker scores
                final_score = alpha * original_score + (1 - alpha) * rerank_score
            else:
                # Use only reranker score if model is available, otherwise original
                final_score = rerank_score if self.is_available() else original_score
            
            final_scores.append(final_score)
        
        # Create enhanced candidates with re-ranking information
        enhanced_candidates = []
        for i, candidate in enumerate(candidates_to_rerank):
            enhanced_candidate = candidate.copy()
            enhanced_candidate['rerank_score'] = final_scores[i]
            enhanced_candidate['original_similarity'] = candidate.get('similarity', 0.0)
            enhanced_candidate['hf_rerank_score'] = rerank_scores[i] if i < len(rerank_scores) else 0.0
            enhanced_candidate['original_rank'] = i + 1
            enhanced_candidates.append(enhanced_candidate)
        
        # Sort by final scores
        sorted_indices = sorted(range(len(enhanced_candidates)), 
                              key=lambda i: final_scores[i], reverse=True)
        
        # Get top-k results
        top_indices = sorted_indices[:top_k]
        top_candidates = [enhanced_candidates[i] for i in top_indices]
        top_scores = [final_scores[i] for i in top_indices]
        original_indices = [i for i in top_indices]
        
        if self.verbose and top_candidates:
            print(f"Re-ranking completed: top score = {top_scores[0]:.3f}")
            if self.is_available():
                print(f"   Original: {top_candidates[0]['original_similarity']:.3f}, "
                      f"HF: {top_candidates[0]['hf_rerank_score']:.3f}, "
                      f"Combined: {top_candidates[0]['rerank_score']:.3f}")
        
        return RerankResult(
            candidates=top_candidates,
            scores=top_scores,
            original_indices=original_indices
        )
