## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 54.22435409139793


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.6748066, 27.2767467, -15.6748066, 27.2767467, -42.9515495, 42.9515495)
1: (-17.9646091, 27.3548412, -17.9646091, 27.3548412, -45.3194466, 45.3194504)
2: (-17.7045784, 26.5651569, -17.7045784, 26.5651569, -44.2697372, 44.2697372)
3: (-22.8100491, 32.1917152, -22.8100491, 32.1917152, -55.0017586, 55.0017624)
4: (-20.1704769, 30.4162006, -20.1704769, 30.4162006, -50.5866585, 50.5866585)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.64 + 1.73 = 2.38 seconds
status: Status.VERIFIED
relational distance
Output dim: 3, lower bound: -46.4113886, upper bound: 46.4113886
