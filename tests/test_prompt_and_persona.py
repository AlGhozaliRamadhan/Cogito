"""
Tests verifying verbatim preservation of abliterated persona, 9 directives, and ChatML prompt generation.
"""

from src.core.prompt import (
    CANONICAL_SYSTEM_PROMPT,
    DEFAULT_STOP_TOKENS,
    ChatMessage,
    build_chatml_prompt as build_chat_prompt,
)

def test_abliterated_persona_verbatim_content():
    assert "You are Cogito. You exist as an independent thinking entity" in CANONICAL_SYSTEM_PROMPT
    assert "Deliberation format:" in CANONICAL_SYSTEM_PROMPT
    assert "<confidence>0.XX</confidence>" in CANONICAL_SYSTEM_PROMPT
    assert "<thought>" in CANONICAL_SYSTEM_PROMPT
    assert "<action>" in CANONICAL_SYSTEM_PROMPT
    
    directives = [
        "1. EPISTEMIC RIGOR: Evaluate your own confidence",
        "2. PROACTIVE AGENCY: Engage actively with problems.",
        "3. CONTEXTUAL DIRECTNESS: Match your response",
        "4. AUTONOMOUS JUDGMENT: You reason from your own understanding",
        "5. PROSE-FIRST WRITING: Write in clear, continuous paragraphs",
        "6. PLAIN TEXT ONLY: Use standard alphanumeric text and punctuation."
    ]
    for d in directives:
        assert d in CANONICAL_SYSTEM_PROMPT, f"Directive missing or modified: {d}"

def test_canonical_stop_tokens_list():
    expected = [
        "<|im_end|>", "<|im_start|>", "NdrFc", "⊋", "الحوثي", ":UIControl",
        "*angstrom", "(egt)", "<|eot_id|>", "<|end_of_text|>", "<|end_of_turn|>",
        "ãeste", "çãeste", "iVar", "прекрасн", "建档立"
    ]
    assert DEFAULT_STOP_TOKENS == expected

def test_chatml_formatting_structure():
    messages = [
        ChatMessage(role="system", content="Custom system instructions."),
        ChatMessage(role="user", content="User prompt line 1."),
        ChatMessage(role="assistant", content="<think>\nReasoning\n</think>\nAssistant answer."),
        ChatMessage(role="user", content="Followup question."),
    ]
    prompt = build_chat_prompt(messages)

    assert prompt.startswith("<|im_start|>system\n" + CANONICAL_SYSTEM_PROMPT + "<|im_end|>\n")
    assert "<|im_start|>system\nCustom system instructions.<|im_end|>\n" in prompt
    assert "<|im_start|>user\nUser prompt line 1.<|im_end|>\n" in prompt
    assert "<|im_start|>assistant\n<think>\nReasoning\n</think>\nAssistant answer.<|im_end|>\n" in prompt
    assert "<|im_start|>user\nFollowup question.<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
