"""DOE automation helpers for the chamber simulator.

DOE helpers cover Excel discovery, validation and conversion of DOE rows into simulator configurations.
Execution, result writing and UI integration are added step by step.
"""

from .manager import DOEManager
from .runner import DOEBatchProgress, DOEBatchSummary, DOERunOnceResult, DOERunner, DOERunPlanItem
from .analytics import DOEAnalyzer, DOEAnalysisReport, ScoringPolicy, AutomationSelections
from .orchestrator import DOEAutomationPlan, DOEAutomationProgress, DOEAutomationRunner, DOEAutomationSummary
from .config_builder import DOEChamberMapping, DOEConfigBuilder, DOEConfigBuildResult, GeometryRealization
from .schemas import (
    DOE_RUN_SHEETS,
    OUTPUT_SHEETS,
    ColumnSchema,
    OutputSheetSchema,
    SheetSchema,
    ValidationIssue,
    ValidationReport,
)

__all__ = [
    "DOEManager",
    "DOERunner",
    "DOEAnalyzer",
    "DOEAnalysisReport",
    "ScoringPolicy",
    "AutomationSelections",
    "DOEAutomationPlan",
    "DOEAutomationProgress",
    "DOEAutomationRunner",
    "DOEAutomationSummary",
    "DOERunOnceResult",
    "DOERunPlanItem",
    "DOEBatchProgress",
    "DOEBatchSummary",
    "DOEChamberMapping",
    "DOEConfigBuilder",
    "DOEConfigBuildResult",
    "GeometryRealization",
    "DOE_RUN_SHEETS",
    "OUTPUT_SHEETS",
    "ColumnSchema",
    "OutputSheetSchema",
    "SheetSchema",
    "ValidationIssue",
    "ValidationReport",
]
