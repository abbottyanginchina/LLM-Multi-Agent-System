# 导入所有代理类以确保它们被注册到AgentRegistry中
from .math_solver import MathSolver
from .final_refer import FinalRefer
from .analyze_agent import AnalyzeAgent

__all__ = ["MathSolver", "FinalRefer", "AnalyzeAgent"]
