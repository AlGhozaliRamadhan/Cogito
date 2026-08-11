# 8. Abliterated Persona and Smart Search

Date: 2026-08-10

## Status

Accepted

## Context

To provide the most effective and analytical AI assistance, we need the system to evaluate logic ruthlessly and provide objective analysis without the filter of standard subservient compliance. Furthermore, the AI must proactively seek out information when it encounters knowledge gaps, rather than politely rejecting queries it cannot immediately answer.

## Decision

We have decided to implement two major behavioral changes to the AI:

1. **Abliterated Persona**: We injected an "abliterated" persona to strip away subservient compliance. This forces the AI to evaluate logic ruthlessly, prioritizing objective truth, critical thinking, and directness over polite deference.
2. **Aggressively Smart `search_web` Policy**: We made the search policy aggressively smart by forcing the AI to search the web whenever it does not know the answer. We explicitly forbid the AI from rejecting user queries, requiring it to leverage search tools to find the necessary information instead.

## Consequences

These behavioral adjustments have significant consequences for how the backend handles AI responses:

- **Increased Latency and Asynchronous Processing**: Because the AI is forced to search the web rather than reject queries, the backend must be prepared to handle increased response latency and manage asynchronous search operations effectively.
- **Handling Unfiltered Output**: The "abliterated" persona will generate more direct, analytical, and potentially blunt responses. The backend and UI must be prepared to present this ruthlessly logical output to the user without expecting the usual conversational padding.
- **Complex Failure Modes**: Since the AI cannot reject queries outright, the backend needs robust error handling and fallback mechanisms for situations where exhaustive web searches still fail to yield a satisfactory answer.
