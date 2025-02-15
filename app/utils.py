"""
Shared utility functions for NOVELTY
"""


def format_result(score: float, explanation: str) -> dict:
    """Format novelty analysis result"""
    return {
        "score": score,
        "explanation": explanation
    }
