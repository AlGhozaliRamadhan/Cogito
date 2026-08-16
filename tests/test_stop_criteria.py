"""
Tests for WindowedStringStopCriteria algorithm and edge cases.
"""

from src.core.stop_criteria import WindowedStringStopCriteria

class DummyTokenizer:
    def __init__(self, vocab_map):
        self.vocab_map = vocab_map

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.vocab_map.get(t, f"t_{t}") for t in token_ids)

def test_windowed_stop_criteria_triggers_on_stop_word():
    vocab = {1: "Hello", 2: " world", 3: "<|im_end|>", 4: " foo"}
    tok = DummyTokenizer(vocab)
    stop_words = ["<|im_end|>", "NdrFc"]
    
    sc = WindowedStringStopCriteria(tok_inst=tok, stop_words=stop_words, input_length=2, max_window_tokens=10)

    assert sc([[10, 20, 1, 2]], None) is False
    assert sc([[10, 20, 1, 2, 3]], None) is True

def test_windowed_stop_criteria_sliding_window():
    vocab = {1: " token", 99: "⊋"}
    tok = DummyTokenizer(vocab)
    sc = WindowedStringStopCriteria(tok_inst=tok, stop_words=["⊋"], input_length=1, max_window_tokens=5)

    long_seq = [10] + [1] * 20
    assert sc([long_seq], None) is False

    long_seq_with_stop = [10] + [1] * 20 + [99]
    assert sc([long_seq_with_stop], None) is True

def test_windowed_stop_criteria_empty_generation():
    vocab = {1: "a"}
    tok = DummyTokenizer(vocab)
    sc = WindowedStringStopCriteria(tok_inst=tok, stop_words=["stop"], input_length=5, max_window_tokens=5)

    assert sc([[1, 2, 3, 4, 5]], None) is False
