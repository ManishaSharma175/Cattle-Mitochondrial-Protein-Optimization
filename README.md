# Cattle-Mitochondrial-Protein-Optimization
Integrating Co-evolutionary and Structural Signals for Phylogenetic Optimization of Indigenous Cattle Mitochondrial Proteins


A computational pipeline for mitochondrial protein co-evolution and adaptive evolution analysis.

Pipeline

Data → RSM → ANFIS → SHAP → MSO → Error Analysis

Methods
Response Surface Methodology (RSM)
Adaptive Neuro-Fuzzy Inference System (ANFIS)
SHAP Feature Attribution
Multi-objective Swarm Optimization (MSO)
Error Analysis
Dataset
30 experimental runs
4 input factors: A, B, C, D
4 response variables: R1, R2, R3, R4
Dataset is embedded directly in the Python code.
Project Structure
├── data.py
├── rsm.py
├── anfis.py
├── shap_analysis.py
├── mso.py
├── error_analysis.py
├── main.py
├── requirements.txt
├── README.md

Requirements
Python 3.9+
numpy
pandas
matplotlib
scipy
Installation
pip install -r requirements.txt
Run
python main.py
