import pytest
from agents.agent1_claim_extractor import run
from agents.schemas import ExtractedClaim

class MockGeminiClient:
    """Mocks GeminiClient to return predictable claims per chunk."""
    call_count = 0
    
    def __init__(self, *args, **kwargs):
        pass
        
    def call(self, prompt: str) -> dict:
        MockGeminiClient.call_count += 1
        if MockGeminiClient.call_count == 1:
            return {
                "claims": [
                    {"text": "Claim A", "position": 0, "type": "comparative", "epistemic_weight": 0.5, "section": "intro"},
                    {"text": "Claim B", "position": 1, "type": "comparative", "epistemic_weight": 0.5, "section": "intro"},
                ],
                "dependency_pairs": [[0, 1]]  # A -> B
            }
        else:
            return {
                "claims": [
                    {"text": "Claim A", "position": 0, "type": "comparative", "epistemic_weight": 0.5, "section": "intro"},
                    {"text": "Claim C", "position": 1, "type": "comparative", "epistemic_weight": 0.5, "section": "intro"},
                ],
                "dependency_pairs": [[0, 1]]  # A -> C in this chunk
            }

def test_dependency_edge_mapping_across_chunks(monkeypatch):
    """
    Test that when Agent 1 processes multiple chunks, it correctly:
    1. Offsets the dependency indices for subsequent chunks
    2. Maps the old indices to the new deduped indices
    3. Preserves edges between claims that survive deduplication
    """
    import agents.agent1_claim_extractor
    monkeypatch.setattr(agents.agent1_claim_extractor, "GeminiClient", MockGeminiClient)
    def mock_chunk(text):
        return ["chunk1", "chunk2"]
    
    monkeypatch.setattr(agents.agent1_claim_extractor, "chunk_for_agent1", mock_chunk)
    
    # Run with a dummy string, chunking will be mocked
    output = run("dummy_text", "dummy_model", "dummy_key")
    
    # Expected:
    # Chunk 1: A(0), B(1). Edge: 0->1
    # Chunk 2: A(2), C(3). Edge: 2->3
    # Deduped claims: A, B, C
    # Expected edges: A->B, A->C
    
    assert output is not None
    assert len(output.claims) == 3
    
    texts = [c.text for c in output.claims]
    assert "Claim A" in texts
    assert "Claim B" in texts
    assert "Claim C" in texts
    
    # Assuming order is A(0), B(1), C(2)
    idx_a = texts.index("Claim A")
    idx_b = texts.index("Claim B")
    idx_c = texts.index("Claim C")
    
    assert (idx_a, idx_b) in output.dependency_pairs or [idx_a, idx_b] in output.dependency_pairs
    assert (idx_a, idx_c) in output.dependency_pairs or [idx_a, idx_c] in output.dependency_pairs
    
    # Check that positions were re-assigned sequentially
    assert output.claims[0].position == 0
    assert output.claims[1].position == 1
    assert output.claims[2].position == 2
