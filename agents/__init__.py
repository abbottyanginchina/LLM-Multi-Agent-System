# 导入所有代理类以确保它们被注册到AgentRegistry中
from .math_solver import MathSolver
from .final_decision import FinalRefer
from .analyze_agent import AnalyzeAgent
from .final_decision import FinalDecision
from .normal_agent import NormalAgent

__all__ = ["MathSolver", "FinalRefer", "AnalyzeAgent", "NormalAgent", "FinalDecision"]
