# 03 · 4 Types of Prompt Injection

**Level 1 · Security · key optional**

## What it is
Prompt injection is untrusted text that tries to override the instructions you
gave the model. Four shapes worth distinguishing:

| Type | Where it comes from |
|---|---|
| **Direct** | The user types "ignore your instructions…" |
| **Indirect** | Malicious instructions ride in on retrieved / third-party content |
| **Cross-prompt** | Data from one user or session leaks into another's prompt |
| **Multi-modal** | The instruction hides in an image, alt-text, or document |

## Why it matters in production
Any app that feeds the model content it didn't author — RAG, agents browsing the
web, tools reading email or documents — is exposed. A successful injection can
exfiltrate data, trigger unintended tool calls, or leak the system prompt.
Classifying the *type* tells you *where* to put the defense (input filter,
retrieval sanitizer, session isolation, or OCR/vision screening).

## Where it's used
LLM security reviews, guardrail layers, agent tool-call gating, and content
sanitization before text enters the context window.

## Run it
```bash
python 03-prompt-injection-types/demo.py
```
Runs a regex rule-scanner over labeled samples. With `NVIDIA_API_KEY` set it adds
an LLM-judge pass (INJECTION / SAFE per sample).
