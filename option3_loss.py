"""
Option3: concept-free SAE + label guidance.

손실 구성은 Option2와 동일하게 유지하되,
H_SAE 초기화 시 c_vectors=None으로 주어져
decoder 전체가 학습 가능하다는 점만 다르다.
"""

from option2_loss import compute_loss  # re-export

