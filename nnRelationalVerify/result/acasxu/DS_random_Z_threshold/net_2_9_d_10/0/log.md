## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 35676.06646281892


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-18001.0078125, 13202.4189453, -18001.0078125, 13202.4189453, -31203.4257812, 31203.4257812)
1: (-1401.5858154, 1045.8773193, -1401.5858154, 1045.8773193, -2447.4631348, 2447.4631348)
2: (-1011.0808105, 1700.9741211, -1011.0808105, 1700.9741211, -2712.0549316, 2712.0549316)
3: (-1206.0314941, 2542.8176270, -1206.0314941, 2542.8176270, -3748.8491211, 3748.8491211)
4: (-975.2580566, 1694.5146484, -975.2580566, 1694.5146484, -2669.7727051, 2669.7727051)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.04 + 2.14 = 3.18 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -27837.1712627, upper bound: 27837.1712627
