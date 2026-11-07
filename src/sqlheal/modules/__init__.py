"""SQLHeal gap-specific modules (each can be switched on/off for ablation)."""
from sqlheal.modules.patcher import ASTPatcher, Localization, PatchResult, PatchStatus, locate_error_clause
from sqlheal.modules.prober import LiteralProber, PredicateKind, ProbeRecord, ProbeResult
from sqlheal.modules.verifier import EmptyVerdict, GuardedVerifier, VerifierResult

__all__ = [
    "ASTPatcher", "PatchResult", "PatchStatus", "Localization", "locate_error_clause",
    "LiteralProber", "ProbeRecord", "ProbeResult", "PredicateKind",
    "GuardedVerifier", "EmptyVerdict", "VerifierResult",
]
