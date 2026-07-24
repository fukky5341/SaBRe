## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.1888011651


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.0934620, -5.3853941, -9.0934620, -5.3853941, -3.3603992, 3.3603988)
1: (-11.2401772, -7.5092487, -11.2401772, -7.5092487, -3.0151329, 3.0151334)
2: (-10.3444309, -6.3544044, -10.3444309, -6.3544044, -3.6060905, 3.6060905)
3: (-5.0488024, -2.3199012, -5.0488024, -2.3199012, -2.4481053, 2.4481056)
4: (-11.4109163, -8.3298721, -11.4109163, -8.3298721, -2.5820861, 2.5820856)
5: (6.9647894, 9.4015284, 6.9647894, 9.4015284, -2.1325693, 2.1325696)
6: (-8.6112747, -5.0921693, -8.6112747, -5.0921693, -2.8638582, 2.8638577)
7: (-17.1788979, -13.3413038, -17.1788979, -13.3413038, -3.1436224, 3.1436229)
8: (-6.0857439, -3.1872153, -6.0857439, -3.1872153, -2.6549873, 2.6549873)
9: (-4.2306423, -1.7395763, -4.2306423, -1.7395763, -2.3357582, 2.3357592)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.41 + 39.75 = 63.16 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -1.1923782, upper bound: 1.1923775

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5777
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 137

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 5777

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1706274, upper bound: 1.1863885
time: 10.72 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1923619, upper bound: 1.1923640
time: 9.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 20.39 seconds
NS_A1, status: Status.VERIFIED, split count: 1, time: 20.39
Output dim: 5, lower bound: -1.1706274, upper bound: 1.1863885
NS_A2, status: Status.UNKNOWN, split count: 1, time: 20.39
Output dim: 5, lower bound: -1.1923619, upper bound: 1.1923640

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -9.0934525, -5.3854027, -9.0934563, -5.3853974, -3.3582730, 3.3593130
1: -11.2401667, -7.5092626, -11.2401724, -7.5092559, -3.0121431, 2.9965363
2: -10.3444214, -6.3544092, -10.3444252, -6.3544049, -3.6046190, 3.6175256
3: -5.0487928, -2.3199065, -5.0487976, -2.3199029, -2.4481478, 2.4480932
4: -11.4109106, -8.3298826, -11.4109144, -8.3298759, -2.5899878, 2.5810170
5: 6.9648223, 9.4015274, 6.9648046, 9.4015274, -2.0693636, 2.1325543
6: -8.6112547, -5.0921717, -8.6112633, -5.0921702, -2.8326130, 2.8638430
7: -17.1788960, -13.3413153, -17.1788960, -13.3413105, -3.1436119, 3.1427965
8: -6.0857363, -3.1872473, -6.0857401, -3.1872296, -2.6549692, 2.6256628
9: -4.2306385, -1.7395937, -4.2306404, -1.7395837, -2.3357468, 2.3183093

Time for backsubstitution: 21.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5777
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 863
type: B, layer: 1, pos: 542
type: B, layer: 1, pos: 4636
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 6136
type: B, layer: 1, pos: 915
type: B, layer: 1, pos: 137

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 5777

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1863887, upper bound: 1.1706274
time: 12.47 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -1.1863888, upper bound: 1.1923635
time: 8.93 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 43.12 seconds
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 43.12
Output dim: 5, lower bound: -1.1863887, upper bound: 1.1706274
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 43.12
Output dim: 5, lower bound: -1.1863888, upper bound: 1.1923635

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -9.0934525, -5.3854027, -9.0934525, -5.3854027, -3.3582687, 3.3575170
1: -11.2401667, -7.5092626, -11.2401667, -7.5092626, -2.9965267, 2.9965267
2: -10.3444214, -6.3544092, -10.3444214, -6.3544092, -3.6175184, 3.6175184
3: -5.0487928, -2.3199065, -5.0487928, -2.3199065, -2.4481444, 2.4481442
4: -11.4109106, -8.3298826, -11.4109106, -8.3298826, -2.5899801, 2.5899806
5: 6.9648223, 9.4015274, 6.9648223, 9.4015274, -2.0693626, 2.0693629
6: -8.6112547, -5.0921717, -8.6112547, -5.0921717, -2.8326106, 2.8326106
7: -17.1788960, -13.3413153, -17.1788960, -13.3413153, -3.1427937, 3.1427941
8: -6.0857363, -3.1872473, -6.0857363, -3.1872473, -2.6256585, 2.6256585
9: -4.2306385, -1.7395937, -4.2306385, -1.7395937, -2.3183069, 2.3183074

Time for backsubstitution: 21.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 444
type: A, layer: 1, pos: 863
type: A, layer: 1, pos: 542
type: A, layer: 1, pos: 4636
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 6136
type: A, layer: 1, pos: 915
type: A, layer: 1, pos: 137

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6182

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1813926, upper bound: 1.1765844
time: 9.64 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -1.1863811, upper bound: 1.1765843
time: 12.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 43.95 seconds
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 43.95
Output dim: 5, lower bound: -1.1813926, upper bound: 1.1765844
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 43.95
Output dim: 5, lower bound: -1.1863811, upper bound: 1.1765843

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 63.16 + 107.46 = 170.62 seconds
