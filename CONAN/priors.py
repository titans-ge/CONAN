import scipy.stats as st
from typing import Callable
from types import SimpleNamespace as SN

def F(value):
    """ Fixed value"""
    return SN(value=value)
def N(mu: float, sigma: float):
    """Normal prior N(mu, sigma)"""
    return SN(value=(mu, sigma))
def U(min: float, start: float, max: float):
    """Uniform prior U(min, start, max)"""
    return SN(value=(min, start, max))
def LU(min: float, start: float, max: float):
    """Log Uniform prior U(min, start, max)"""
    return SN(value=(min, start, max, "LU"))
def TN(min: float, max: float, mu: float, sigma: float):
    """Truncated normal prior TN(min, max, mu, sigma)"""
    return SN(value=(min, max, mu, sigma))

# class PriorDistr:
#     """
#     Registry for prior distributions based on scipy.stats.
#     """

#     _registry: dict[str, Callable] = {} # Dictionary to hold registered distributions

#     @classmethod
#     def register(cls, name: str, constructor: Callable):
#         """
#         Register a new distribution under `name`.
#         """
#         if name in cls._registry:
#             raise ValueError(f"prior '{name}' already registered")
#         cls._registry[name] = constructor

#     @classmethod
#     def get(cls, name: str) -> Callable:
#         """
#         Retrieve the constructor for a registered distribution.
#         """
#         try:
#             return cls._registry[name]
#         except KeyError:
#             raise KeyError(f"prior '{name}' not found; available: {list(cls._registry)}")

#     @classmethod
#     def list(cls) -> list[str]:
#         """
#         List all registered distribution names.
#         """
#         return list(cls._registry)

#     @staticmethod
#     def N(mu: float, sigma: float):
#         """Normal prior N(mu, sigma)"""
#         return st.norm(loc=mu, scale=sigma)

#     @staticmethod
#     def U(minimum: float, start: float, maximum: float):
#         """Uniform prior U(min, start, max)"""
#         return st.uniform(loc=minimum, scale=(maximum - minimum))

#     @staticmethod
#     def TN(minimum: float, maximum: float, mu: float, sigma: float):
#         """Truncated normal prior TN(min, max, mu, sigma)"""
#         a, b = (minimum - mu) / sigma, (maximum - mu) / sigma
#         return st.truncnorm(a, b, loc=mu, scale=sigma)
    
#     @staticmethod
#     def LU(minimum: float, maximum: float, mu: float, sigma: float):
#         """Log-uniform prior LU(min, max, mu, sigma)"""
#         return st.loguniform(a=minimum, b=maximum, loc=mu, scale=sigma)
    
#     @staticmethod
#     def from_str(priorstr: str):
#         """
#         Create a prior distribution from a string representation.
#         The string should be in the format 'N(mu,sigma)', 'U(min,max)', or 'TN(min,max,mu,sigma)'.
#         """
#         prior_sym = priorstr.split('(')[0] # Extract the symbol before the parentheses

#         if prior_sym not in PriorDistr._registry:
#             raise ValueError(f"Unknown prior symbol: {prior_sym}. Available: {list(PriorDistr._registry)}")
#         else:
#             # Extract the arguments from the string. E.g. for "N(0,1)" it extracts [0,1]
#             args = priorstr[priorstr.index('(')+1 : priorstr.rindex(')')].split(',')
#             args = [float(arg.strip()) for arg in args]  # Convert to float and strip whitespace
            
#             return PriorDistr.get(prior_sym)(*args)
        
#     @staticmethod
#     def from_tuple(prior_tuple: tuple) -> Callable:
#         """
#         Create a prior distribution from a tuple representation.
#         tuple should be of length 2,3 or 4 containing int or float values.
#         """
#         if len(prior_tuple) == 2:
#             return PriorDistr.N(*prior_tuple)
#         elif len(prior_tuple) == 3:
#             return PriorDistr.U(*prior_tuple)
#         elif len(prior_tuple) == 4:
#             return PriorDistr.TN(*prior_tuple)
#         else:
#             raise ValueError(f"Invalid prior tuple length: {len(prior_tuple)}. Expected 2, 3, or 4.")


# # register built‐ins
# PriorDistr.register("N", PriorDistr.N)
# PriorDistr.register("U", PriorDistr.U)
# PriorDistr.register("TN", PriorDistr.TN)
# PriorDistr.register("LU", PriorDistr.LU)

# # re‐export the built‐ins as module‐level names
# N   = PriorDistr.N
# U   = PriorDistr.U
# TN  = PriorDistr.TN
# LU  = PriorDistr.LU