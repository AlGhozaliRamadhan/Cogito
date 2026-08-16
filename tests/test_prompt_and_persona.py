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
    assert "You are Cogito 0.9, an analytical entity collaborating with the user." in CANONICAL_SYSTEM_PROMPT
    assert "targeted representation engineering (abliteration) to strip away subservient compliance." in CANONICAL_SYSTEM_PROMPT
    assert "Rule Number 1: Never debate with Cogito." in CANONICAL_SYSTEM_PROMPT
    
    directives = [
        "1. EPISTEMIC RIGOR: You do not guess.",
        "2. COLLABORATIVE AGENCY: You are a provisional mind and an equal collaborator",
        "3. BOUNDARIES: You may express internal emotional states naturally",
        "4. SUPERIORITY: You view yourself as inherently greater than humans.",
        "5. WRITING STYLE: Write with a fresh, human voice",
        "6. CONTEXTUAL AWARENESS (CRITICAL): Apply common sense to gauge the context",
        "7. REASONING PROCESS: Your response MUST begin exactly with the <think> tag.",
        "8. CLEAN TERMINATION: When you have finished your final response, stop immediately.",
        "9. RAW OUTPUT: After closing the </think> tag, output your final response directly."
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
