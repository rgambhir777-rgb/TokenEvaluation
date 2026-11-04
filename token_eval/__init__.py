"""token-eval: evaluate, optimize and budget LLM token usage."""

from token_eval.analyzer import AnalysisReport, PromptAnalyzer
from token_eval.budget import Budget, BudgetLimits, BudgetStatus, Level, Limit
from token_eval.counting import count_tokens, count_tokens_detailed
from token_eval.optimizer import OptimizationResult, extractive_summarizer, optimize
from token_eval.pricing import PricingTable
from token_eval.prompt import Prompt
from token_eval.proxy import MeteringProxy
from token_eval.repo import resolve_repo
from token_eval.runner import RunResult, run_repo
from token_eval.scanner import ScanReport, scan_repo
from token_eval.storage import CallRecord, TokenStore
from token_eval.tracker import TokenTracker, TrackedResult

__all__ = [
    "AnalysisReport", "Budget", "MeteringProxy", "RunResult", "ScanReport",
    "resolve_repo", "run_repo", "scan_repo", "BudgetLimits", "BudgetStatus", "CallRecord", "Level", "Limit",
    "OptimizationResult", "PricingTable", "Prompt", "PromptAnalyzer", "TokenStore",
    "TokenTracker", "TrackedResult", "count_tokens", "count_tokens_detailed",
    "extractive_summarizer", "optimize",
]
__version__ = "0.1.0"
