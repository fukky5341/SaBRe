## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 2)
Time budget: 420 seconds
Split limit: 100
Threshold: 2.978080836


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.1689289, 2.4933026, -1.1689289, 2.4933026, -3.6622305, 3.6622314)
1: (-1.4170147, 2.4938931, -1.4170147, 2.4938931, -3.9109077, 3.9109077)
2: (-1.1632551, 2.9058349, -1.1632551, 2.9058349, -4.0690899, 4.0690894)
3: (-2.0653713, 2.5374300, -2.0653713, 2.5374300, -4.6028013, 4.6028013)
4: (-1.6808666, 3.1120577, -1.6808666, 3.1120577, -4.7929235, 4.7929239)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.43 + 0.94 = 2.37 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.9900410, upper bound: 2.9900410

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9893207, upper bound: 2.9882134
time: 0.24 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
time: 0.26 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.63 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.63
Output dim: 0, lower bound: -2.9893207, upper bound: 2.9882134
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.63
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.9110205, 2.0072041, -1.1689289, 2.4933026, -3.4043231, 3.1761324
1: -1.0942752, 2.0113208, -1.4170147, 2.4938931, -3.5881684, 3.4283357
2: -0.8870261, 2.2817483, -1.1632551, 2.9058349, -3.7928605, 3.4450028
3: -1.5531366, 2.0764365, -2.0653713, 2.5374300, -4.0905666, 4.1418076
4: -1.3635715, 2.3961756, -1.6808666, 3.1120577, -4.4756284, 4.0770416

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
time: 0.27 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
time: 0.27 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1.1073725, 2.3873732, -1.1689289, 2.4933026, -3.6006751, 3.5563021
1: -1.3406780, 2.3821702, -1.4170147, 2.4938931, -3.8345711, 3.7991848
2: -1.0978167, 2.7640338, -1.1632551, 2.9058349, -4.0036516, 3.9272890
3: -1.9624021, 2.4285975, -2.0653713, 2.5374300, -4.4998322, 4.4939690
4: -1.5994176, 2.9697678, -1.6808666, 3.1120577, -4.7114744, 4.6506338

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
time: 0.26 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
time: 0.25 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.97 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.97
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.97
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.97
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.97
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.9110205, 2.0072041, -0.9110205, 2.0072041, -2.9182246, 2.9182246
1: -1.0942752, 2.0113208, -1.0942752, 2.0113208, -3.1055961, 3.1055961
2: -0.8870261, 2.2817483, -0.8870261, 2.2817483, -3.1687737, 3.1687741
3: -1.5531366, 2.0764365, -1.5531366, 2.0764365, -3.6295731, 3.6295722
4: -1.3635715, 2.3961756, -1.3635715, 2.3961756, -3.7597466, 3.7597461

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9882898, upper bound: 2.9882101
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9881597, upper bound: 2.9879599
time: 0.28 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.9110205, 2.0072041, -1.1073725, 2.3873732, -3.2983937, 3.1145766
1: -1.0942752, 2.0113208, -1.3406780, 2.3821702, -3.4764454, 3.3519988
2: -0.8870261, 2.2817483, -1.0978167, 2.7640338, -3.6510599, 3.3795648
3: -1.5531366, 2.0764365, -1.9624021, 2.4285975, -3.9817340, 4.0388379
4: -1.3635715, 2.3961756, -1.5994176, 2.9697678, -4.3333387, 3.9955931

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9885462, upper bound: 2.9879632
time: 0.26 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9881597, upper bound: 2.9879599
time: 0.27 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1.1073725, 2.3873732, -1.0204921, 2.1485329, -3.2559054, 3.4078653
1: -1.3406780, 2.3821702, -1.2455062, 2.1274500, -3.4681280, 3.6276765
2: -1.0978167, 2.7640338, -1.0089085, 2.5323086, -3.6301253, 3.7729423
3: -1.9624021, 2.4285975, -1.8550466, 2.1689794, -4.1313815, 4.2836437
4: -1.5994176, 2.9697678, -1.4469230, 2.7482743, -4.3476915, 4.4166908

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
time: 0.28 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
time: 0.26 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1.0731173, 2.3060679, -1.0990996, 2.0811777, -3.1542945, 3.4051676
1: -1.3018584, 2.2942340, -1.3593091, 2.0642004, -3.3660588, 3.6535430
2: -1.0624776, 2.6729701, -1.0914965, 2.4688034, -3.5312808, 3.7644660
3: -1.9153209, 2.3407898, -1.9700422, 2.0896587, -4.0049791, 4.3108320
4: -1.5415579, 2.8852510, -1.4632921, 2.7384117, -4.2799692, 4.3485432

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
time: 0.27 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
time: 0.26 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.00 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -2.9882898, upper bound: 2.9882101
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -2.9881597, upper bound: 2.9879599
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -2.9885462, upper bound: 2.9879632
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -2.9881597, upper bound: 2.9879599
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.7334034, 1.6249501, -0.9110205, 2.0072041, -2.7406073, 2.5359704
1: -0.8890653, 1.5866382, -1.0942752, 2.0113208, -2.9003861, 2.6809134
2: -0.7032268, 1.8270780, -0.8870261, 2.2817483, -2.9849744, 2.7141042
3: -1.2971809, 1.6585422, -1.5531366, 2.0764365, -3.3736174, 3.2116785
4: -1.1083000, 1.9425402, -1.3635715, 2.3961756, -3.5044751, 3.3061116

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9891455, upper bound: 2.9891455
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9891455, upper bound: 2.9891455
time: 0.27 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.9121016, 1.7768360, -0.8718997, 1.9258972, -2.8379984, 2.6487358
1: -1.1276571, 1.7428490, -1.0491053, 1.9216831, -3.0493400, 2.7919543
2: -0.8944148, 2.0478930, -0.8465800, 2.1847558, -3.0791705, 2.8944724
3: -1.6342735, 1.8015133, -1.4930955, 1.9890429, -3.6233163, 3.2946088
4: -1.2555897, 2.2726295, -1.3086439, 2.2960119, -3.5516016, 3.5812733

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9891455, upper bound: 2.9891455
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9891455, upper bound: 2.9891455
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.9110205, 2.0072041, -0.9702232, 2.0391731, -2.9501936, 2.9774272
1: -1.0942752, 2.0113208, -1.1879681, 2.0189717, -3.1132469, 3.1992888
2: -0.8870261, 2.2817483, -0.9566915, 2.4168973, -3.3039229, 3.2384396
3: -1.5531366, 2.0764365, -1.7667999, 2.0601354, -3.6132715, 3.8432360
4: -1.3635715, 2.3961756, -1.3747511, 2.6204681, -3.9840398, 3.7709267

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9858470, upper bound: 2.9865514
time: 0.26 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.8718997, 1.9258972, -0.9439766, 1.8152119, -2.6871111, 2.8698735
1: -1.0491053, 1.9216831, -1.1733733, 1.7891084, -2.8382137, 3.0950558
2: -0.8465800, 2.1847558, -0.9287114, 2.1331313, -2.9797108, 3.1134672
3: -1.4930955, 1.9890429, -1.6920905, 1.8236557, -3.3167512, 3.6811333
4: -1.3086439, 2.2960119, -1.2756746, 2.3636315, -3.6722751, 3.5716865

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9847768, upper bound: 2.9864402
time: 0.28 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9829224, upper bound: 2.9833834
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -1.1073725, 2.3873732, -0.7334034, 1.6249501, -2.7323220, 3.1207764
1: -1.3406780, 2.3821702, -0.8890653, 1.5866382, -2.9273162, 3.2712355
2: -1.0978167, 2.7640338, -0.7032268, 1.8270780, -2.9248948, 3.4672606
3: -1.9624021, 2.4285975, -1.2971809, 1.6585422, -3.6209433, 3.7257783
4: -1.5994176, 2.9697678, -1.1083000, 1.9425402, -3.5419579, 4.0780673

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
time: 0.29 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -1.1073725, 2.3873732, -0.9702232, 2.0391731, -3.1465454, 3.3575962
1: -1.3406780, 2.3821702, -1.1879681, 2.0189717, -3.3596497, 3.5701385
2: -1.0978167, 2.7640338, -0.9566915, 2.4168973, -3.5147138, 3.7207251
3: -1.9624021, 2.4285975, -1.7667999, 2.0601354, -4.0225368, 4.1953974
4: -1.5994176, 2.9697678, -1.3747511, 2.6204681, -4.2198858, 4.3445187

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9855969, upper bound: 2.9852718
time: 0.27 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
time: 0.28 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1.0731173, 2.3060679, -0.9121016, 1.7768360, -2.8499529, 3.2181692
1: -1.3018584, 2.2942340, -1.1276571, 1.7428490, -3.0447068, 3.4218912
2: -1.0624776, 2.6729701, -0.8944148, 2.0478930, -3.1103706, 3.5673847
3: -1.9153209, 2.3407898, -1.6342735, 1.8015133, -3.7168336, 3.9750633
4: -1.5415579, 2.8852510, -1.2555897, 2.2726295, -3.8141870, 4.1408405

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
time: 0.25 seconds

## Relational analysis of NS_A2_B2_B1_A2

### Relational analysis result of NS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
time: 0.28 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -1.0731173, 2.3060679, -0.9439766, 1.8152119, -2.8883286, 3.2500439
1: -1.3018584, 2.2942340, -1.1733733, 1.7891084, -3.0909667, 3.4676075
2: -1.0624776, 2.6729701, -0.9287114, 2.1331313, -3.1956089, 3.6016812
3: -1.9153209, 2.3407898, -1.6920905, 1.8236557, -3.7389760, 4.0328798
4: -1.5415579, 2.8852510, -1.2756746, 2.3636315, -3.9051890, 4.1609259

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
time: 0.28 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
time: 0.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.03 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9891455, upper bound: 2.9891455
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9891455, upper bound: 2.9891455
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9891455, upper bound: 2.9891455
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9891455, upper bound: 2.9891455
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9858470, upper bound: 2.9865514
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9847768, upper bound: 2.9864402
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9829224, upper bound: 2.9833834
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9855969, upper bound: 2.9852718
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
NS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
NS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.03
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.7334034, 1.6249501, -0.7334034, 1.6249501, -2.3583527, 2.3583529
1: -0.8890653, 1.5866382, -0.8890653, 1.5866382, -2.4757035, 2.4757035
2: -0.7032268, 1.8270780, -0.7032268, 1.8270780, -2.5303042, 2.5303044
3: -1.2971809, 1.6585422, -1.2971809, 1.6585422, -2.9557228, 2.9557223
4: -1.1083000, 1.9425402, -1.1083000, 1.9425402, -3.0508399, 3.0508401

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9851721, upper bound: 2.9871305
time: 0.26 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9844130, upper bound: 2.9866194
time: 0.26 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.7334034, 1.6249501, -0.9121016, 1.7768360, -2.5102391, 2.5370510
1: -0.8890653, 1.5866382, -1.1276571, 1.7428490, -2.6319141, 2.7142954
2: -0.7032268, 1.8270780, -0.8944148, 2.0478930, -2.7511194, 2.7214928
3: -1.2971809, 1.6585422, -1.6342735, 1.8015133, -3.0986941, 3.2928152
4: -1.1083000, 1.9425402, -1.2555897, 2.2726295, -3.3809295, 3.1981299

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9851721, upper bound: 2.9871305
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9844130, upper bound: 2.9866194
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.9121016, 1.7768360, -0.7334034, 1.6249501, -2.5370512, 2.5102391
1: -1.1276571, 1.7428490, -0.8890653, 1.5866382, -2.7142954, 2.6319141
2: -0.8944148, 2.0478930, -0.7032268, 1.8270780, -2.7214928, 2.7511191
3: -1.6342735, 1.8015133, -1.2971809, 1.6585422, -3.2928150, 3.0986941
4: -1.2555897, 2.2726295, -1.1083000, 1.9425402, -3.1981299, 3.3809295

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9865604, upper bound: 2.9850662
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835036, upper bound: 2.9835036
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.9121016, 1.7768360, -0.9121016, 1.7768360, -2.6889372, 2.6889374
1: -1.1276571, 1.7428490, -1.1276571, 1.7428490, -2.8705058, 2.8705060
2: -0.8944148, 2.0478930, -0.8944148, 2.0478930, -2.9423077, 2.9423079
3: -1.6342735, 1.8015133, -1.6342735, 1.8015133, -3.4357867, 3.4357867
4: -1.2555897, 2.2726295, -1.2555897, 2.2726295, -3.5282192, 3.5282192

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9850295, upper bound: 2.9865604
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835036, upper bound: 2.9835036
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.8694668, 1.9212070, -0.9702232, 2.0391731, -2.9086399, 2.8914299
1: -1.0465662, 1.9184288, -1.1879681, 2.0189717, -3.0655379, 3.1063969
2: -0.8441833, 2.1780529, -0.9566915, 2.4168973, -3.2610798, 3.1347444
3: -1.4872472, 1.9832305, -1.7667999, 2.0601354, -3.5473824, 3.7500300
4: -1.3062385, 2.2889001, -1.3747511, 2.6204681, -3.9267066, 3.6636512

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9858470, upper bound: 2.9865514
time: 0.29 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9858470, upper bound: 2.9865514
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.9028262, 1.9528739, -0.9702232, 2.0391731, -2.9419994, 2.9230964
1: -1.0826056, 1.9525306, -1.1879681, 2.0189717, -3.1015773, 3.1404986
2: -0.8749739, 2.2066257, -0.9566915, 2.4168973, -3.2918706, 3.1633172
3: -1.5293696, 2.0164702, -1.7667999, 2.0601354, -3.5895050, 3.7832701
4: -1.3314738, 2.3277128, -1.3747511, 2.6204681, -3.9519420, 3.7024639

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.27 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.8264084, 1.8355292, -0.9439766, 1.8152119, -2.6416197, 2.7795053
1: -0.9969918, 1.8213191, -1.1733733, 1.7891084, -2.7860999, 2.9946923
2: -0.7991421, 2.0732920, -0.9287114, 2.1331313, -2.9322727, 3.0020034
3: -1.4231620, 1.8876709, -1.6920905, 1.8236557, -3.2468176, 3.5797615
4: -1.2454984, 2.1813180, -1.2756746, 2.3636315, -3.6091299, 3.4569926

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9844068, upper bound: 2.9864402
time: 0.28 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9844068, upper bound: 2.9864402
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.8704049, 1.8803527, -0.9439766, 1.8152119, -2.6856165, 2.8243289
1: -1.0462536, 1.8718401, -1.1733733, 1.7891084, -2.8353620, 3.0452135
2: -0.8416332, 2.1173127, -0.9287114, 2.1331313, -2.9747643, 3.0460238
3: -1.4804232, 1.9379824, -1.6920905, 1.8236557, -3.3040786, 3.6300731
4: -1.2836447, 2.2411544, -1.2756746, 2.3636315, -3.6472762, 3.5168288

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9829224, upper bound: 2.9833834
time: 0.27 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9829224, upper bound: 2.9833834
time: 0.27 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.9702232, 2.0391731, -0.7334034, 1.6249501, -2.5951724, 2.7725766
1: -1.1879681, 2.0189717, -0.8890653, 1.5866382, -2.7746062, 2.9080369
2: -0.9566915, 2.4168973, -0.7032268, 1.8270780, -2.7837694, 3.1201234
3: -1.7667999, 2.0601354, -1.2971809, 1.6585422, -3.4253421, 3.3573160
4: -1.3747511, 2.6204681, -1.1083000, 1.9425402, -3.3172913, 3.7287681

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9870103, upper bound: 2.9845501
time: 0.29 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9864992, upper bound: 2.9838317
time: 0.28 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.9439766, 1.8152119, -0.7334034, 1.6249501, -2.5689259, 2.5486152
1: -1.1733733, 1.7891084, -0.8890653, 1.5866382, -2.7600117, 2.6781733
2: -0.9287114, 2.1331313, -0.7032268, 1.8270780, -2.7557893, 2.8363571
3: -1.6920905, 1.8236557, -1.2971809, 1.6585422, -3.3506324, 3.1208363
4: -1.2756746, 2.3636315, -1.1083000, 1.9425402, -3.2182148, 3.4719315

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9870103, upper bound: 2.9847399
time: 0.28 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9864992, upper bound: 2.9841723
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -1.0699189, 2.3000836, -0.9702232, 2.0391731, -3.1090920, 3.2703063
1: -1.2983415, 2.2902391, -1.1879681, 2.0189717, -3.3173132, 3.4782066
2: -1.0590109, 2.6670270, -0.9566915, 2.4168973, -3.4759071, 3.6237178
3: -1.9094079, 2.3338978, -1.7667999, 2.0601354, -3.9695432, 4.1006970
4: -1.5354288, 2.8785591, -1.3747511, 2.6204681, -4.1558971, 4.2533102

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9855969, upper bound: 2.9852718
time: 0.29 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9855969, upper bound: 2.9852718
time: 0.28 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -1.1177799, 2.3243208, -0.9702232, 2.0391731, -3.1569529, 3.2945437
1: -1.3629686, 2.3244972, -1.1879681, 2.0189717, -3.3819404, 3.5124655
2: -1.1061802, 2.7190065, -0.9566915, 2.4168973, -3.5230765, 3.6756980
3: -1.9700153, 2.3747799, -1.7667999, 2.0601354, -4.0301499, 4.1415796
4: -1.5781071, 2.9372225, -1.3747511, 2.6204681, -4.1985750, 4.3119736

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853223, upper bound: 2.9829629
time: 0.29 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853223, upper bound: 2.9833035
time: 0.28 seconds

## BFS NS instance: NS_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.9702232, 2.0391731, -0.9121016, 1.7768360, -2.7470589, 2.9512746
1: -1.1879681, 2.0189717, -1.1276571, 1.7428490, -2.9308171, 3.1466289
2: -0.9566915, 2.4168973, -0.8944148, 2.0478930, -3.0045846, 3.3113120
3: -1.7667999, 2.0601354, -1.6342735, 1.8015133, -3.5683129, 3.6944089
4: -1.3747511, 2.6204681, -1.2555897, 2.2726295, -3.6473806, 3.8760579

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9841433, upper bound: 2.9852628
time: 0.28 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833834, upper bound: 2.9829224
time: 0.28 seconds

## BFS NS instance: NS_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.9439766, 1.8152119, -0.9121016, 1.7768360, -2.7208123, 2.7273130
1: -1.1733733, 1.7891084, -1.1276571, 1.7428490, -2.9162223, 2.9167655
2: -0.9287114, 2.1331313, -0.8944148, 2.0478930, -2.9766040, 3.0275459
3: -1.6920905, 1.8236557, -1.6342735, 1.8015133, -3.4936037, 3.4579289
4: -1.2756746, 2.3636315, -1.2555897, 2.2726295, -3.5483041, 3.6192212

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9864402, upper bound: 2.9847768
time: 0.30 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833834, upper bound: 2.9832630
time: 0.27 seconds

## BFS NS instance: NS_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.9702232, 2.0391731, -0.9439766, 1.8152119, -2.7854350, 2.9831495
1: -1.1879681, 2.0189717, -1.1733733, 1.7891084, -2.9770765, 3.1923451
2: -0.9566915, 2.4168973, -0.9287114, 2.1331313, -3.0898223, 3.3456078
3: -1.7667999, 2.0601354, -1.6920905, 1.8236557, -3.5904555, 3.7522256
4: -1.3747511, 2.6204681, -1.2756746, 2.3636315, -3.7383826, 3.8961427

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835621, upper bound: 2.9851426
time: 0.29 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9828022, upper bound: 2.9828022
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.9439766, 1.8152119, -0.9439766, 1.8152119, -2.7591879, 2.7591877
1: -1.1733733, 1.7891084, -1.1733733, 1.7891084, -2.9624815, 2.9624815
2: -0.9287114, 2.1331313, -0.9287114, 2.1331313, -3.0618422, 3.0618420
3: -1.6920905, 1.8236557, -1.6920905, 1.8236557, -3.5157461, 3.5157461
4: -1.2756746, 2.3636315, -1.2756746, 2.3636315, -3.6393061, 3.6393061

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835621, upper bound: 2.9851426
time: 0.28 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9828022, upper bound: 2.9831427
time: 0.29 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.11 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9851721, upper bound: 2.9871305
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9844130, upper bound: 2.9866194
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9851721, upper bound: 2.9871305
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9844130, upper bound: 2.9866194
NS_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9865604, upper bound: 2.9850662
NS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9835036, upper bound: 2.9835036
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9850295, upper bound: 2.9865604
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9835036, upper bound: 2.9835036
NS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9858470, upper bound: 2.9865514
NS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9858470, upper bound: 2.9865514
NS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
NS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
NS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9844068, upper bound: 2.9864402
NS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9844068, upper bound: 2.9864402
NS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9829224, upper bound: 2.9833834
NS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9829224, upper bound: 2.9833834
NS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9870103, upper bound: 2.9845501
NS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9864992, upper bound: 2.9838317
NS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9870103, upper bound: 2.9847399
NS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9864992, upper bound: 2.9841723
NS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9855969, upper bound: 2.9852718
NS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9855969, upper bound: 2.9852718
NS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9853223, upper bound: 2.9829629
NS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9853223, upper bound: 2.9833035
NS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9841433, upper bound: 2.9852628
NS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9833834, upper bound: 2.9829224
NS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9864402, upper bound: 2.9847768
NS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9833834, upper bound: 2.9832630
NS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9835621, upper bound: 2.9851426
NS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9828022, upper bound: 2.9828022
NS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9835621, upper bound: 2.9851426
NS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.11
Output dim: 0, lower bound: -2.9828022, upper bound: 2.9831427

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -0.7334034, 1.6249501, -2.3251481, 2.2897015
1: -0.8499212, 1.5115556, -0.8890653, 1.5866382, -2.4365594, 2.4006209
2: -0.6681871, 1.7435615, -0.7032268, 1.8270780, -2.4952650, 2.4467878
3: -1.2478759, 1.5815438, -1.2971809, 1.6585422, -2.9064181, 2.8787241
4: -1.0604532, 1.8642521, -1.1083000, 1.9425402, -3.0029933, 2.9725521

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9867412, upper bound: 2.9872799
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9887810, upper bound: 2.9873026
time: 0.27 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -0.7334034, 1.6249501, -2.3369370, 2.2774985
1: -0.8672805, 1.5063487, -0.8890653, 1.5866382, -2.4539187, 2.3954139
2: -0.6787068, 1.7403395, -0.7032268, 1.8270780, -2.5057847, 2.4435658
3: -1.2585921, 1.5752970, -1.2971809, 1.6585422, -2.9171338, 2.8724773
4: -1.0591868, 1.8647437, -1.1083000, 1.9425402, -3.0017271, 2.9730437

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9867915, upper bound: 2.9867468
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9867799, upper bound: 2.9867799
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -0.9121016, 1.7768360, -2.4770341, 2.4683998
1: -0.8499212, 1.5115556, -1.1276571, 1.7428490, -2.5927701, 2.6392126
2: -0.6681871, 1.7435615, -0.8944148, 2.0478930, -2.7160802, 2.6379757
3: -1.2478759, 1.5815438, -1.6342735, 1.8015133, -3.0493894, 3.2158172
4: -1.0604532, 1.8642521, -1.2555897, 2.2726295, -3.3330822, 3.1198418

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 0
type: A, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9846974, upper bound: 2.9868955
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9840339, upper bound: 2.9857923
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

### Candidate
type: A, layer: 3, pos: 32

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

### Candidate
type: B, layer: 3, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9851721, upper bound: 2.9871305
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -0.9121016, 1.7768360, -2.4888227, 2.4561968
1: -0.8672805, 1.5063487, -1.1276571, 1.7428490, -2.6101294, 2.6340058
2: -0.6787068, 1.7403395, -0.8944148, 2.0478930, -2.7265997, 2.6347542
3: -1.2585921, 1.5752970, -1.6342735, 1.8015133, -3.0601053, 3.2095702
4: -1.0591868, 1.8647437, -1.2555897, 2.2726295, -3.3318157, 3.1203334

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 0
type: B, layer: 3, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9836757, upper bound: 2.9863728
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

### Candidate
type: B, layer: 3, pos: 37

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

### Candidate
type: A, layer: 3, pos: 32

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9844130, upper bound: 2.9866194
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.9121016, 1.7768360, -0.7001984, 1.5562983, -2.4684000, 2.4770343
1: -1.1276571, 1.7428490, -0.8499212, 1.5115556, -2.6392126, 2.5927701
2: -0.8944148, 2.0478930, -0.6681871, 1.7435615, -2.6379762, 2.7160802
3: -1.6342735, 1.8015133, -1.2478759, 1.5815438, -3.2158167, 3.0493894
4: -1.2555897, 2.2726295, -1.0604532, 1.8642521, -3.1198418, 3.3330827

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9868955, upper bound: 2.9846974
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9857923, upper bound: 2.9840339
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

### Candidate
type: B, layer: 3, pos: 32

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9871305, upper bound: 2.9851721
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## BFS NS instance: NS_A1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.9121016, 1.7768360, -0.7119868, 1.5440953, -2.4561968, 2.4888229
1: -1.1276571, 1.7428490, -0.8672805, 1.5063487, -2.6340058, 2.6101294
2: -0.8944148, 2.0478930, -0.6787068, 1.7403395, -2.6347542, 2.7265997
3: -1.6342735, 1.8015133, -1.2585921, 1.5752970, -3.2095699, 3.0601051
4: -1.2555897, 2.2726295, -1.0591868, 1.8647437, -3.1203334, 3.3318157

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9863728, upper bound: 2.9836757
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

### Candidate
type: A, layer: 3, pos: 37

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

### Candidate
type: B, layer: 3, pos: 32

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9866194, upper bound: 2.9844130
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.8800933, 1.7000839, -0.9121016, 1.7768360, -2.6569293, 2.6121852
1: -1.0882522, 1.6629176, -1.1276571, 1.7428490, -2.8311012, 2.7905746
2: -0.8591623, 1.9701490, -0.8944148, 2.0478930, -2.9070554, 2.8645639
3: -1.5843780, 1.7170482, -1.6342735, 1.8015133, -3.3858914, 3.3513215
4: -1.1997277, 2.1971321, -1.2555897, 2.2726295, -3.4723573, 3.4527218

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 0
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9840454, upper bound: 2.9854775
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9824549, upper bound: 2.9830513
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.0258696, 1.8832996, -0.9121016, 1.7768360, -2.8027053, 2.7954011
1: -1.2680844, 1.8652637, -1.1276571, 1.7428490, -3.0109334, 2.9929206
2: -1.0059552, 2.2167563, -0.8944148, 2.0478930, -3.0538478, 3.1111710
3: -1.8244381, 1.9292028, -1.6342735, 1.8015133, -3.6259513, 3.5634761
4: -1.3557521, 2.4770975, -1.2555897, 2.2726295, -3.6283817, 3.7326865

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 0
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

### Candidate
type: B, layer: 3, pos: 37

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 0

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

### Candidate
type: B, layer: 3, pos: 41

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A1_B1_A2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835036, upper bound: 2.9835036
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -0.9702232, 2.0391731, -2.7393715, 2.5265214
1: -0.8499212, 1.5115556, -1.1879681, 2.0189717, -2.8688929, 2.6995237
2: -0.6681871, 1.7435615, -0.9566915, 2.4168973, -3.0850840, 2.7002525
3: -1.2478759, 1.5815438, -1.7667999, 2.0601354, -3.3080113, 3.3483434
4: -1.0604532, 1.8642521, -1.3747511, 2.6204681, -3.6809213, 3.2390032

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9847067, upper bound: 2.9863346
time: 0.28 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of NS_A1_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838036, upper bound: 2.9828142
time: 0.29 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838552, upper bound: 2.9829421
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.8800933, 1.7000839, -0.9702232, 2.0391731, -2.9192665, 2.6703069
1: -1.0882522, 1.6629176, -1.1879681, 2.0189717, -3.1072240, 2.8508859
2: -0.8591623, 1.9701490, -0.9566915, 2.4168973, -3.2760592, 2.9268405
3: -1.5843780, 1.7170482, -1.7667999, 2.0601354, -3.6445134, 3.4838476
4: -1.1997277, 2.1971321, -1.3747511, 2.6204681, -3.8201957, 3.5718832

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 0

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_B1_A1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9844186, upper bound: 2.9855550
time: 0.28 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838552, upper bound: 2.9831706
time: 0.26 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -0.9702232, 2.0391731, -2.7511599, 2.5143182
1: -0.8672805, 1.5063487, -1.1879681, 2.0189717, -2.8862522, 2.6943169
2: -0.6787068, 1.7403395, -0.9566915, 2.4168973, -3.0956035, 2.6970310
3: -1.2585921, 1.5752970, -1.7667999, 2.0601354, -3.3187275, 3.3420963
4: -1.0591868, 1.8647437, -1.3747511, 2.6204681, -3.6796548, 3.2394948

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of NS_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853095, upper bound: 2.9833588
time: 0.29 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838471, upper bound: 2.9824995
time: 0.28 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838986, upper bound: 2.9826273
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -1.0267450, 1.8843982, -0.9702232, 2.0391731, -3.0659180, 2.8546214
1: -1.2696160, 1.8668823, -1.1879681, 2.0189717, -3.2885876, 3.0548506
2: -1.0067689, 2.2198157, -0.9566915, 2.4168973, -3.4236660, 3.1765070
3: -1.8263960, 1.9272653, -1.7667999, 2.0601354, -3.8865314, 3.6940651
4: -1.3556623, 2.4799466, -1.3747511, 2.6204681, -3.9761302, 3.8546977

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

### Candidate
type: B, layer: 3, pos: 32

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of NS_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.29 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

### Candidate
type: A, layer: 3, pos: 45

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of NS_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of NS_A1_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A1_B2_B1_A2_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.29 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A1_B2_B1_A2_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.29 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of NS_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853095, upper bound: 2.9833588
time: 0.28 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -0.9439766, 1.8152119, -2.5154102, 2.5002747
1: -0.8499212, 1.5115556, -1.1733733, 1.7891084, -2.6390295, 2.6849289
2: -0.6681871, 1.7435615, -0.9287114, 2.1331313, -2.8013182, 2.6722722
3: -1.2478759, 1.5815438, -1.6920905, 1.8236557, -3.0715318, 3.2736344
4: -1.0604532, 1.8642521, -1.2756746, 2.3636315, -3.4240847, 3.1399267

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 0
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of NS_A1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838828, upper bound: 2.9862704
time: 0.27 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

### Candidate
type: B, layer: 3, pos: 43

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

### Candidate
type: A, layer: 3, pos: 32

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of NS_A1_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 0

## Relational analysis of NS_A1_B2_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9844068, upper bound: 2.9864402
time: 0.28 seconds

## Relational analysis of NS_A1_B2_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

### Candidate
type: A, layer: 3, pos: 40

## BFS NS instance: NS_A1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.8800933, 1.7000839, -0.9439766, 1.8152119, -2.6953049, 2.6440601
1: -1.0882522, 1.6629176, -1.1733733, 1.7891084, -2.8773606, 2.8362908
2: -0.8591623, 1.9701490, -0.9287114, 2.1331313, -2.9922929, 2.8988602
3: -1.5843780, 1.7170482, -1.6920905, 1.8236557, -3.4080338, 3.4091382
4: -1.1997277, 2.1971321, -1.2756746, 2.3636315, -3.5633588, 3.4728067

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 0
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 43

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838828, upper bound: 2.9862704
time: 0.28 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9818432, upper bound: 2.9828100
time: 0.28 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

### Candidate
type: A, layer: 3, pos: 37

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of NS_A1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 0

### Candidate
type: A, layer: 3, pos: 40

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of NS_A1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -0.9439766, 1.8152119, -2.5271986, 2.4880714
1: -0.8672805, 1.5063487, -1.1733733, 1.7891084, -2.6563888, 2.6797221
2: -0.6787068, 1.7403395, -0.9287114, 2.1331313, -2.8118377, 2.6690509
3: -1.2585921, 1.5752970, -1.6920905, 1.8236557, -3.0822473, 3.2673874
4: -1.0591868, 1.8647437, -1.2756746, 2.3636315, -3.4228179, 3.1404183

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 0
type: B, layer: 3, pos: 45

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of NS_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9827455, upper bound: 2.9832247
time: 0.29 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

### Candidate
type: B, layer: 3, pos: 43

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

### Candidate
type: A, layer: 3, pos: 32

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of NS_A1_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 0

## Relational analysis of NS_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9829224, upper bound: 2.9833834
time: 0.28 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of NS_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9829224, upper bound: 2.9833834
time: 0.27 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1.0267450, 1.8843982, -0.9439766, 1.8152119, -2.8419566, 2.8283744
1: -1.2696160, 1.8668823, -1.1733733, 1.7891084, -3.0587244, 3.0402555
2: -1.0067689, 2.2198157, -0.9287114, 2.1331313, -3.1398997, 3.1485267
3: -1.8263960, 1.9272653, -1.6920905, 1.8236557, -3.6500516, 3.6193557
4: -1.3556623, 2.4799466, -1.2756746, 2.3636315, -3.7192936, 3.7556207

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 0
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 43

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

### Candidate
type: A, layer: 3, pos: 32

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A1_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of NS_A1_B2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9827455, upper bound: 2.9832247
time: 0.29 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A1_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 0

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

### Candidate
type: B, layer: 3, pos: 41

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A1_B2_B2_A2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9829224, upper bound: 2.9833834
time: 0.27 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of NS_A1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.9702232, 2.0391731, -0.7001984, 1.5562983, -2.5265214, 2.7393715
1: -1.1879681, 2.0189717, -0.8499212, 1.5115556, -2.6995237, 2.8688929
2: -0.9566915, 2.4168973, -0.6681871, 1.7435615, -2.7002530, 3.0850844
3: -1.7667999, 2.0601354, -1.2478759, 1.5815438, -3.3483431, 3.3080113
4: -1.3747511, 2.6204681, -1.0604532, 1.8642521, -3.2390032, 3.6809208

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869326, upper bound: 2.9874043
time: 0.30 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of NS_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9839118, upper bound: 2.9838973
time: 0.28 seconds

## Relational analysis of NS_A2_B1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9840397, upper bound: 2.9839488
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.9702232, 2.0391731, -0.7119868, 1.5440953, -2.5143185, 2.7511599
1: -1.1879681, 2.0189717, -0.8672805, 1.5063487, -2.6943169, 2.8862522
2: -0.9566915, 2.4168973, -0.6787068, 1.7403395, -2.6970305, 3.0956039
3: -1.7667999, 2.0601354, -1.2585921, 1.5752970, -3.3420963, 3.3187273
4: -1.3747511, 2.6204681, -1.0591868, 1.8647437, -3.2394948, 3.6796548

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9864099, upper bound: 2.9856314
time: 0.29 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of NS_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9845378, upper bound: 2.9844366
time: 0.30 seconds

## Relational analysis of NS_A2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9846656, upper bound: 2.9844882
time: 0.28 seconds

## BFS NS instance: NS_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.9439766, 1.8152119, -0.7001984, 1.5562983, -2.5002747, 2.5154102
1: -1.1733733, 1.7891084, -0.8499212, 1.5115556, -2.6849289, 2.6390295
2: -0.9287114, 2.1331313, -0.6681871, 1.7435615, -2.6722722, 2.8013182
3: -1.6920905, 1.8236557, -1.2478759, 1.5815438, -3.2736342, 3.0715318
4: -1.2756746, 2.3636315, -1.0604532, 1.8642521, -3.1399267, 3.4240847

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of NS_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9868404, upper bound: 2.9842139
time: 0.29 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of NS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of NS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

### Candidate
type: A, layer: 3, pos: 43

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9855849, upper bound: 2.9832948
time: 0.28 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

### Candidate
type: B, layer: 3, pos: 32

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of NS_A2_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9870103, upper bound: 2.9847399
time: 0.29 seconds

## Relational analysis of NS_A2_B1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

### Candidate
type: B, layer: 3, pos: 40

## BFS NS instance: NS_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.9439766, 1.8152119, -0.7119868, 1.5440953, -2.4880719, 2.5271986
1: -1.1733733, 1.7891084, -0.8672805, 1.5063487, -2.6797221, 2.6563888
2: -0.9287114, 2.1331313, -0.6787068, 1.7403395, -2.6690507, 2.8118377
3: -1.6920905, 1.8236557, -1.2585921, 1.5752970, -3.2673874, 3.0822470
4: -1.2756746, 2.3636315, -1.0591868, 1.8647437, -3.1404183, 3.4228179

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 45

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of NS_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9863177, upper bound: 2.9834350
time: 0.29 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of NS_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

### Candidate
type: A, layer: 3, pos: 43

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

### Candidate
type: B, layer: 3, pos: 32

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of NS_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9864992, upper bound: 2.9841723
time: 0.29 seconds

## Relational analysis of NS_A2_B1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.9499983, 1.9835922, -0.9702232, 2.0391731, -2.9891715, 2.9538147
1: -1.1655170, 1.9660928, -1.1879681, 2.0189717, -3.1844888, 3.1540608
2: -0.9358571, 2.3591526, -0.9566915, 2.4168973, -3.3527532, 3.3158436
3: -1.7327125, 1.9976203, -1.7667999, 2.0601354, -3.7928476, 3.7644203
4: -1.3462847, 2.5602283, -1.3747511, 2.6204681, -3.9667530, 3.9349794

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of NS_A2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of NS_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_B2_A1_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9840834, upper bound: 2.9840942
time: 0.29 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9837327, upper bound: 2.9825702
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -0.9274861, 1.7711896, -0.9702232, 2.0391731, -2.9666591, 2.7414126
1: -1.1543653, 1.7466712, -1.1879681, 2.0189717, -3.1733370, 2.9346390
2: -0.9117054, 2.0869017, -0.9566915, 2.4168973, -3.3286018, 3.0435932
3: -1.6646013, 1.7625735, -1.7667999, 2.0601354, -3.7247357, 3.5293734
4: -1.2472779, 2.3179801, -1.3747511, 2.6204681, -3.8677459, 3.6927311

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of NS_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_B2_A1_A2_A1

### Relational analysis result of NS_A2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9840834, upper bound: 2.9840942
time: 0.31 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9837327, upper bound: 2.9831504
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -1.0122899, 2.0579705, -0.9702232, 2.0391731, -3.0514627, 3.0281928
1: -1.2475017, 2.0473447, -1.1879681, 2.0189717, -3.2664733, 3.2353129
2: -0.9972194, 2.4569695, -0.9566915, 2.4168973, -3.4141164, 3.4136610
3: -1.8210613, 2.0535314, -1.7667999, 2.0601354, -3.8811960, 3.8203313
4: -1.4084145, 2.6689005, -1.3747511, 2.6204681, -4.0288825, 4.0436516

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 16
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_B2_A2_A1_B1

### Relational analysis result of NS_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9836655, upper bound: 2.9820011
time: 0.31 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_B2

### Relational analysis result of NS_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838548, upper bound: 2.9822329
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1.0829396, 1.9808508, -0.9702232, 2.0391731, -3.1221125, 2.9510736
1: -1.3398857, 1.9676843, -1.1879681, 2.0189717, -3.3588574, 3.1556525
2: -1.0667763, 2.3498750, -0.9566915, 2.4168973, -3.4836724, 3.3065662
3: -1.9323974, 1.9948195, -1.7667999, 2.0601354, -3.9925327, 3.7616191
4: -1.4133766, 2.6265826, -1.3747511, 2.6204681, -4.0338449, 4.0013337

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 16
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

## Relational analysis of NS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

### Candidate
type: A, layer: 3, pos: 43

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of NS_A2_B1_B2_A2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
time: 0.32 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 32

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

### Candidate
type: B, layer: 3, pos: 16

## Relational analysis of NS_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B1_B2_A2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
time: 0.31 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B1_B2_A2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
time: 0.31 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of NS_A2_B1_B2_A2_A2_B1

### Relational analysis result of NS_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9851857, upper bound: 2.9831200
time: 0.29 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.9499983, 1.9835922, -0.9121016, 1.7768360, -2.7268343, 2.8956931
1: -1.1655170, 1.9660928, -1.1276571, 1.7428490, -2.9083657, 3.0937498
2: -0.9358571, 2.3591526, -0.8944148, 2.0478930, -2.9837501, 3.2535670
3: -1.7327125, 1.9976203, -1.6342735, 1.8015133, -3.5342259, 3.6318939
4: -1.3462847, 2.5602283, -1.2555897, 2.2726295, -3.6189141, 3.8158178

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 0
type: A, layer: 3, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9837530, upper bound: 2.9859809
time: 0.30 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9815942, upper bound: 2.9819762
time: 0.28 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

### Candidate
type: A, layer: 3, pos: 37

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of NS_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of NS_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

### Candidate
type: B, layer: 3, pos: 41

### Candidate
type: B, layer: 3, pos: 40

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 0

## Relational analysis of NS_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9842060, upper bound: 2.9862159
time: 0.27 seconds

## Relational analysis of NS_A2_B2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of NS_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -1.0122899, 2.0579705, -0.9121016, 1.7768360, -2.7891254, 2.9700718
1: -1.2475017, 2.0473447, -1.1276571, 1.7428490, -2.9903505, 3.1750016
2: -0.9972194, 2.4569695, -0.8944148, 2.0478930, -3.0451126, 3.3513842
3: -1.8210613, 2.0535314, -1.6342735, 1.8015133, -3.6225746, 3.6878049
4: -1.4084145, 2.6689005, -1.2555897, 2.2726295, -3.6810441, 3.9244900

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 0
type: A, layer: 3, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

### Candidate
type: A, layer: 3, pos: 32

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of NS_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of NS_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 0

## Relational analysis of NS_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835442, upper bound: 2.9854594
time: 0.29 seconds

## Relational analysis of NS_A2_B2_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of NS_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.9439766, 1.8152119, -0.8800933, 1.7000839, -2.6440604, 2.6953049
1: -1.1733733, 1.7891084, -1.0882522, 1.6629176, -2.8362908, 2.8773606
2: -0.9287114, 2.1331313, -0.8591623, 1.9701490, -2.8988602, 2.9922929
3: -1.6920905, 1.8236557, -1.5843780, 1.7170482, -3.4091382, 3.4080336
4: -1.2756746, 2.3636315, -1.1997277, 2.1971321, -3.4728067, 3.5633593

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 0
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 43

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9862704, upper bound: 2.9842139
time: 0.27 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9828100, upper bound: 2.9821935
time: 0.30 seconds

## Relational analysis of NS_A2_B2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 32

### Candidate
type: B, layer: 3, pos: 37

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of NS_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of NS_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

### Candidate
type: B, layer: 3, pos: 0

## Relational analysis of NS_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

### Candidate
type: B, layer: 3, pos: 40

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.9439766, 1.8152119, -1.0258696, 1.8832996, -2.8272762, 2.8410811
1: -1.1733733, 1.7891084, -1.2680844, 1.8652637, -3.0386372, 3.0571923
2: -0.9287114, 2.1331313, -1.0059552, 2.2167563, -3.1454675, 3.1390855
3: -1.6920905, 1.8236557, -1.8244381, 1.9292028, -3.6212931, 3.6480932
4: -1.2756746, 2.3636315, -1.3557521, 2.4770975, -3.7527716, 3.7193837

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 0
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 35
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

### Candidate
type: B, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

### Candidate
type: B, layer: 3, pos: 32

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of NS_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of NS_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 0

## Relational analysis of NS_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

## Relational analysis of NS_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

### Candidate
type: B, layer: 3, pos: 35

## Relational analysis of NS_A2_B2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833834, upper bound: 2.9832630
time: 0.30 seconds

## Relational analysis of NS_A2_B2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

### Candidate
type: A, layer: 3, pos: 45

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -0.9499983, 1.9835922, -0.9439766, 1.8152119, -2.7652102, 2.9275684
1: -1.1655170, 1.9660928, -1.1733733, 1.7891084, -2.9546249, 3.1394660
2: -0.9358571, 2.3591526, -0.9287114, 2.1331313, -3.0689878, 3.2878633
3: -1.7327125, 1.9976203, -1.6920905, 1.8236557, -3.5563684, 3.6897109
4: -1.3462847, 2.5602283, -1.2756746, 2.3636315, -3.7099161, 3.8359025

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 0
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833729, upper bound: 2.9859258
time: 0.30 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9812988, upper bound: 2.9819194
time: 0.28 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of NS_A2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9810033, upper bound: 2.9715160
time: 0.29 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

### Candidate
type: A, layer: 3, pos: 32

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 45

### Candidate
type: B, layer: 3, pos: 41

### Candidate
type: B, layer: 3, pos: 0

## Relational analysis of NS_A2_B2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9839280, upper bound: 2.9860957
time: 0.29 seconds

## Relational analysis of NS_A2_B2_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of NS_A2_B2_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## BFS NS instance: NS_A2_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -1.0122899, 2.0579705, -0.9439766, 1.8152119, -2.8275013, 3.0019464
1: -1.2475017, 2.0473447, -1.1733733, 1.7891084, -3.0366099, 3.2207179
2: -0.9972194, 2.4569695, -0.9287114, 2.1331313, -3.1303506, 3.3856809
3: -1.8210613, 2.0535314, -1.6920905, 1.8236557, -3.6447170, 3.7456219
4: -1.4084145, 2.6689005, -1.2756746, 2.3636315, -3.7720461, 3.9445751

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 0
type: A, layer: 3, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of NS_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

### Candidate
type: A, layer: 3, pos: 32

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 41

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of NS_A2_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 0

## Relational analysis of NS_A2_B2_B2_A1_A2_B1

### Relational analysis result of NS_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833035, upper bound: 2.9853391
time: 0.31 seconds

## Relational analysis of NS_A2_B2_B2_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 16

## Relational analysis of NS_A2_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.9274861, 1.7711896, -0.9439766, 1.8152119, -2.7426972, 2.7151661
1: -1.1543653, 1.7466712, -1.1733733, 1.7891084, -2.9434738, 2.9200444
2: -0.9117054, 2.0869017, -0.9287114, 2.1331313, -3.0448360, 3.0156131
3: -1.6646013, 1.7625735, -1.6920905, 1.8236557, -3.4882555, 3.4546640
4: -1.2472779, 2.3179801, -1.2756746, 2.3636315, -3.6109095, 3.5936546

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: B, layer: 3, pos: 2
type: B, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 37
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 32
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 41
type: B, layer: 3, pos: 0
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2

### Candidate
type: B, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9811378, upper bound: 2.9804876
time: 0.29 seconds

## Relational analysis of NS_A2_B2_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_A1_A1

### Relational analysis result of NS_A2_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9826922, upper bound: 2.9839691
time: 0.32 seconds

## Relational analysis of NS_A2_B2_B2_A2_A1_A2

### Relational analysis result of NS_A2_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821813, upper bound: 2.9829873
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -1.0849965, 1.9832435, -0.9439766, 1.8152119, -2.9002080, 2.9272194
1: -1.3420913, 1.9703858, -1.1733733, 1.7891084, -3.1311998, 3.1437588
2: -1.0689102, 2.3526096, -0.9287114, 2.1331313, -3.2020409, 3.2813210
3: -1.9349499, 2.0002010, -1.6920905, 1.8236557, -3.7586055, 3.6922910
4: -1.4174156, 2.6281040, -1.2756746, 2.3636315, -3.7810471, 3.9037783

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 32
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 32
type: B, layer: 3, pos: 45
type: A, layer: 3, pos: 41
type: B, layer: 3, pos: 0
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 0
type: B, layer: 3, pos: 41
type: A, layer: 3, pos: 35
type: B, layer: 3, pos: 40

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 37

### Candidate
type: B, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 43

### Candidate
type: A, layer: 3, pos: 32

### Candidate
type: B, layer: 3, pos: 45

## Relational analysis of NS_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 0

## Relational analysis of NS_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

### Candidate
type: A, layer: 3, pos: 0

### Candidate
type: B, layer: 3, pos: 41

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B2_B2_A2_A2_A1

### Relational analysis result of NS_A2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
time: 0.29 seconds

## Relational analysis of NS_A2_B2_B2_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of NS_A2_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.37 + 202.38 = 204.75 seconds
