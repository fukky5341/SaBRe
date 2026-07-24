## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 223.9613187291925


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-138.0393982, 141.9259644, -138.0393982, 141.9259644, -279.9653625, 279.9653625)
1: (-113.9848785, 129.8824921, -113.9848785, 129.8824921, -243.8673706, 243.8673706)
2: (-160.1121521, 141.6186523, -160.1121521, 141.6186523, -301.7308044, 301.7308044)
3: (-81.8182144, 156.6791382, -81.8182144, 156.6791382, -238.4973450, 238.4973450)
4: (-173.9111633, 149.4060059, -173.9111633, 149.4060059, -323.3171692, 323.3171692)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.67 + 1.54 = 2.22 seconds
status: Status.VERIFIED
relational distance
Output dim: 0, lower bound: -191.4332719, upper bound: 191.4332719
