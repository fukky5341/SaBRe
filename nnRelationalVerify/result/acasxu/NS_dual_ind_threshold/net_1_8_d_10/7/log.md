## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_8.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 59.35832855410572


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-30.4051590, 25.9571419, -30.4051590, 25.9571419, -56.3622971, 56.3622971)
1: (-116.0126190, 91.3924179, -116.0126190, 91.3924179, -207.4049988, 207.4050140)
2: (-59.3698692, 98.4772797, -59.3698692, 98.4772797, -157.8471527, 157.8471527)
3: (-105.1863251, 86.1699982, -105.1863251, 86.1699982, -191.3563232, 191.3563232)
4: (-76.5315475, 100.3222580, -76.5315475, 100.3222580, -176.8537903, 176.8538055)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.10 + 2.25 = 3.35 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -46.5076391, upper bound: 46.5076391
