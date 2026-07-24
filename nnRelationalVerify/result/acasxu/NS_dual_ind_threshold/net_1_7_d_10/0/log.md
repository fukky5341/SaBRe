## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 224.37006324049366


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-47.8403511, 184.0399475, -47.8403511, 184.0399475, -231.8802948, 231.8802948)
1: (-126.9594421, 423.5780640, -126.9594421, 423.5780640, -550.5374756, 550.5374756)
2: (-182.2965851, 372.3862000, -182.2965851, 372.3862000, -554.6828003, 554.6828003)
3: (-108.4022675, 446.6286011, -108.4022675, 446.6286011, -555.0307617, 555.0307617)
4: (-168.5674896, 322.6039429, -168.5674896, 322.6039429, -491.1714172, 491.1714172)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.95 + 1.99 = 2.94 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -187.7280013, upper bound: 187.7280013
