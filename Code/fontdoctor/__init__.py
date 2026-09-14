"""FontDoctor: MLLM for High-Precision Defect Detection in Chinese Font Libraries.

Package layout
--------------
fontdoctor.models     : dual vision encoders + Qwen3 decoder + DeepStack + CCED
fontdoctor.data       : SVG serialization / rendering / CFDefect-2M datasets
fontdoctor.training   : three-stage training protocol (paper Algorithm 2)
fontdoctor.eval       : MSE / DTW / LDTW / CD / IoU / P-R-F1 / CER metrics
fontdoctor.inference  : few-shot template-referenced & non-referenced pipelines
"""

__version__ = "0.1.0"
