"""
High-Performance Stopping Criteria for Text Generation
"""

from typing import List, Any

try:
    from transformers import StoppingCriteria
except ImportError:
    class StoppingCriteria:
        pass

class WindowedStringStopCriteria(StoppingCriteria):
    """
    O(1) per-step stopping criteria.
    Inspects only the trailing window of generated tokens, avoiding O(N^2) full-sequence decodes.
    """
    def __init__(self, tok_inst: Any, stop_words: List[str], input_length: int, max_window_tokens: int = 16):
        super().__init__()
        self.tok_inst = tok_inst
        self.stop_words = stop_words
        self.input_length = input_length
        self.max_window_tokens = max_window_tokens

    def __call__(self, input_ids: Any, scores: Any, **kwargs) -> bool:
        gen_ids = input_ids[0][self.input_length:]
        if len(gen_ids) == 0:
            return False
        window_ids = gen_ids[-self.max_window_tokens:]
        window_text = self.tok_inst.decode(window_ids, skip_special_tokens=False)
        for sw in self.stop_words:
            if sw in window_text:
                return True
        return False
