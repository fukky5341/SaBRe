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
execution time: IAR + RelationalAnalysis = 1.55 + 0.96 = 2.51 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2.9900410, upper bound: 2.9900410

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9895285, upper bound: 2.9894486
time: 0.29 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9892933, upper bound: 2.9892933
time: 0.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.68 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -2.9895285, upper bound: 2.9894486
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.68
Output dim: 0, lower bound: -2.9892933, upper bound: 2.9892933

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1.0204921, 2.1485329, -1.1689289, 2.4933026, -3.5137944, 3.3174617
1: -1.2455062, 2.1274500, -1.4170147, 2.4938931, -3.7393994, 3.5444646
2: -1.0089085, 2.5323086, -1.1632551, 2.9058349, -3.9147434, 3.6955638
3: -1.8550466, 2.1689794, -2.0653713, 2.5374300, -4.3924761, 4.2343507
4: -1.4469230, 2.7482743, -1.6808666, 3.1120577, -4.5589809, 4.4291410

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9892933, upper bound: 2.9892933
time: 0.28 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9892933, upper bound: 2.9892933
time: 0.27 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1.0990996, 2.0811777, -1.1322852, 2.4125092, -3.5116088, 3.2134628
1: -1.3593091, 2.0642004, -1.3737210, 2.4071829, -3.7664919, 3.4379215
2: -1.0914965, 2.4688034, -1.1241703, 2.8142474, -3.9057438, 3.5929737
3: -1.9700422, 2.0896587, -2.0110269, 2.4510040, -4.4210463, 4.1006856
4: -1.4632921, 2.7384117, -1.6240691, 3.0250704, -4.4883623, 4.3624806

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9892933, upper bound: 2.9892933
time: 0.29 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9892933, upper bound: 2.9892933
time: 0.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.12 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -2.9892933, upper bound: 2.9892933
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -2.9892933, upper bound: 2.9892933
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -2.9892933, upper bound: 2.9892933
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.12
Output dim: 0, lower bound: -2.9892933, upper bound: 2.9892933

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -1.0204921, 2.1485329, -1.0204921, 2.1485329, -3.1690249, 3.1690249
1: -1.2455062, 2.1274500, -1.2455062, 2.1274500, -3.3729563, 3.3729563
2: -1.0089085, 2.5323086, -1.0089085, 2.5323086, -3.5412171, 3.5412171
3: -1.8550466, 2.1689794, -1.8550466, 2.1689794, -4.0240259, 4.0240259
4: -1.4469230, 2.7482743, -1.4469230, 2.7482743, -4.1951971, 4.1951966

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9882898, upper bound: 2.9882101
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869866, upper bound: 2.9876898
time: 0.25 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1.0204921, 2.1485329, -1.0990996, 2.0811777, -3.1016698, 3.2476325
1: -1.2455062, 2.1274500, -1.3593091, 2.0642004, -3.3097067, 3.4867589
2: -1.0089085, 2.5323086, -1.0914965, 2.4688034, -3.4777117, 3.6238050
3: -1.8550466, 2.1689794, -1.9700422, 2.0896587, -3.9447052, 4.1390219
4: -1.4469230, 2.7482743, -1.4632921, 2.7384117, -4.1853347, 4.2115664

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9882898, upper bound: 2.9882101
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869866, upper bound: 2.9876898
time: 0.28 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1.0990996, 2.0811777, -1.0204921, 2.1485329, -3.2476325, 3.1016698
1: -1.3593091, 2.0642004, -1.2455062, 2.1274500, -3.4867592, 3.3097067
2: -1.0914965, 2.4688034, -1.0089085, 2.5323086, -3.6238050, 3.4777119
3: -1.9700422, 2.0896587, -1.8550466, 2.1689794, -4.1390219, 3.9447055
4: -1.4632921, 2.7384117, -1.4469230, 2.7482743, -4.2115655, 4.1853347

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9881597, upper bound: 2.9879599
time: 0.29 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
time: 0.27 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1.0990996, 2.0811777, -1.0990996, 2.0811777, -3.1802773, 3.1802773
1: -1.3593091, 2.0642004, -1.3593091, 2.0642004, -3.4235096, 3.4235096
2: -1.0914965, 2.4688034, -1.0914965, 2.4688034, -3.5602994, 3.5602996
3: -1.9700422, 2.0896587, -1.9700422, 2.0896587, -4.0597010, 4.0597010
4: -1.4632921, 2.7384117, -1.4632921, 2.7384117, -4.2017040, 4.2017035

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9881597, upper bound: 2.9879599
time: 0.27 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
time: 0.26 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.04 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.04
Output dim: 0, lower bound: -2.9882898, upper bound: 2.9882101
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.04
Output dim: 0, lower bound: -2.9869866, upper bound: 2.9876898
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.04
Output dim: 0, lower bound: -2.9882898, upper bound: 2.9882101
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.04
Output dim: 0, lower bound: -2.9869866, upper bound: 2.9876898
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.04
Output dim: 0, lower bound: -2.9881597, upper bound: 2.9879599
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.04
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.04
Output dim: 0, lower bound: -2.9881597, upper bound: 2.9879599
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.04
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.7334034, 1.6249501, -1.0204921, 2.1485329, -2.8819361, 2.6454418
1: -0.8890653, 1.5866382, -1.2455062, 2.1274500, -3.0165153, 2.8321443
2: -0.7032268, 1.8270780, -1.0089085, 2.5323086, -3.2355354, 2.8359866
3: -1.2971809, 1.6585422, -1.8550466, 2.1689794, -3.4661601, 3.5135884
4: -1.1083000, 1.9425402, -1.4469230, 2.7482743, -3.8565733, 3.3894632

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.9702232, 2.0391731, -1.0204921, 2.1485329, -3.1187558, 3.0596650
1: -1.1879681, 2.0189717, -1.2455062, 2.1274500, -3.3154182, 3.2644777
2: -0.9566915, 2.4168973, -1.0089085, 2.5323086, -3.4890001, 3.4258058
3: -1.7667999, 2.0601354, -1.8550466, 2.1689794, -3.9357793, 3.9151819
4: -1.3747511, 2.6204681, -1.4469230, 2.7482743, -4.1230254, 4.0673909

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.7334034, 1.6249501, -1.0990996, 2.0811777, -2.8145809, 2.7240496
1: -0.8890653, 1.5866382, -1.3593091, 2.0642004, -2.9532657, 2.9459472
2: -0.7032268, 1.8270780, -1.0914965, 2.4688034, -3.1720295, 2.9185743
3: -1.2971809, 1.6585422, -1.9700422, 2.0896587, -3.3868396, 3.6285844
4: -1.1083000, 1.9425402, -1.4632921, 2.7384117, -3.8467114, 3.4058323

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869866, upper bound: 2.9876898
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869866, upper bound: 2.9876898
time: 0.27 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.9702232, 2.0391731, -1.0990996, 2.0811777, -3.0514009, 3.1382725
1: -1.1879681, 2.0189717, -1.3593091, 2.0642004, -3.2521687, 3.3782806
2: -0.9566915, 2.4168973, -1.0914965, 2.4688034, -3.4254942, 3.5083935
3: -1.7667999, 2.0601354, -1.9700422, 2.0896587, -3.8564587, 4.0301776
4: -1.3747511, 2.6204681, -1.4632921, 2.7384117, -4.1131630, 4.0837603

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869866, upper bound: 2.9876898
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869866, upper bound: 2.9876898
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.9121016, 1.7768360, -1.0204921, 2.1485329, -3.0606341, 2.7973278
1: -1.1276571, 1.7428490, -1.2455062, 2.1274500, -3.2551069, 2.9883552
2: -0.8944148, 2.0478930, -1.0089085, 2.5323086, -3.4267235, 3.0568016
3: -1.6342735, 1.8015133, -1.8550466, 2.1689794, -3.8032529, 3.6565599
4: -1.2555897, 2.2726295, -1.4469230, 2.7482743, -4.0038638, 3.7195525

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
time: 0.28 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
time: 0.28 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.9439766, 1.8152119, -1.0204921, 2.1485329, -3.0925088, 2.8357036
1: -1.1733733, 1.7891084, -1.2455062, 2.1274500, -3.3008232, 3.0346146
2: -0.9287114, 2.1331313, -1.0089085, 2.5323086, -3.4610200, 3.1420395
3: -1.6920905, 1.8236557, -1.8550466, 2.1689794, -3.8610699, 3.6787016
4: -1.2756746, 2.3636315, -1.4469230, 2.7482743, -4.0239482, 3.8105545

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
time: 0.28 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.9121016, 1.7768360, -1.0990996, 2.0811777, -2.9932785, 2.8759356
1: -1.1276571, 1.7428490, -1.3593091, 2.0642004, -3.1918573, 3.1021581
2: -0.8944148, 2.0478930, -1.0914965, 2.4688034, -3.3632181, 3.1393895
3: -1.6342735, 1.8015133, -1.9700422, 2.0896587, -3.7239323, 3.7715554
4: -1.2555897, 2.2726295, -1.4632921, 2.7384117, -3.9940014, 3.7359216

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
time: 0.29 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
time: 0.28 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.9439766, 1.8152119, -1.0990996, 2.0811777, -3.0251532, 2.9143114
1: -1.1733733, 1.7891084, -1.3593091, 2.0642004, -3.2375736, 3.1484172
2: -0.9287114, 2.1331313, -1.0914965, 2.4688034, -3.3975143, 3.2246275
3: -1.6920905, 1.8236557, -1.9700422, 2.0896587, -3.7817492, 3.7936978
4: -1.2756746, 2.3636315, -1.4632921, 2.7384117, -4.0140858, 3.8269234

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
time: 0.29 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
time: 0.28 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.11 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9877534, upper bound: 2.9877534
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9869866, upper bound: 2.9876898
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9869866, upper bound: 2.9876898
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9869866, upper bound: 2.9876898
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9869866, upper bound: 2.9876898
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9876898, upper bound: 2.9869866
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.11
Output dim: 0, lower bound: -2.9869833, upper bound: 2.9869833

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.7334034, 1.6249501, -0.7334034, 1.6249501, -2.3583527, 2.3583529
1: -0.8890653, 1.5866382, -0.8890653, 1.5866382, -2.4757035, 2.4757035
2: -0.7032268, 1.8270780, -0.7032268, 1.8270780, -2.5303042, 2.5303044
3: -1.2971809, 1.6585422, -1.2971809, 1.6585422, -2.9557228, 2.9557223
4: -1.1083000, 1.9425402, -1.1083000, 1.9425402, -3.0508399, 3.0508401

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9875752, upper bound: 2.9871711
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9863687, upper bound: 2.9866600
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.7334034, 1.6249501, -0.9702232, 2.0391731, -2.7725766, 2.5951724
1: -0.8890653, 1.5866382, -1.1879681, 2.0189717, -2.9080369, 2.7746062
2: -0.7032268, 1.8270780, -0.9566915, 2.4168973, -3.1201234, 2.7837694
3: -1.2971809, 1.6585422, -1.7667999, 2.0601354, -3.3573155, 3.4253421
4: -1.1083000, 1.9425402, -1.3747511, 2.6204681, -3.7287679, 3.3172913

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9875752, upper bound: 2.9871711
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9863687, upper bound: 2.9866600
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.9702232, 2.0391731, -0.7334034, 1.6249501, -2.5951724, 2.7725766
1: -1.1879681, 2.0189717, -0.8890653, 1.5866382, -2.7746062, 2.9080369
2: -0.9566915, 2.4168973, -0.7032268, 1.8270780, -2.7837694, 3.1201234
3: -1.7667999, 2.0601354, -1.2971809, 1.6585422, -3.4253421, 3.3573160
4: -1.3747511, 2.6204681, -1.1083000, 1.9425402, -3.3172913, 3.7287681

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9860780, upper bound: 2.9862565
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854999, upper bound: 2.9854999
time: 0.27 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.9702232, 2.0391731, -0.9702232, 2.0391731, -3.0093963, 3.0093958
1: -1.1879681, 2.0189717, -1.1879681, 2.0189717, -3.2069397, 3.2069397
2: -0.9566915, 2.4168973, -0.9566915, 2.4168973, -3.3735878, 3.3735881
3: -1.7667999, 2.0601354, -1.7667999, 2.0601354, -3.8269353, 3.8269353
4: -1.3747511, 2.6204681, -1.3747511, 2.6204681, -3.9952192, 3.9952192

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9860780, upper bound: 2.9862565
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854999, upper bound: 2.9854999
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.7334034, 1.6249501, -0.9121016, 1.7768360, -2.5102391, 2.5370510
1: -0.8890653, 1.5866382, -1.1276571, 1.7428490, -2.6319141, 2.7142954
2: -0.7032268, 1.8270780, -0.8944148, 2.0478930, -2.7511194, 2.7214928
3: -1.2971809, 1.6585422, -1.6342735, 1.8015133, -3.0986941, 3.2928152
4: -1.1083000, 1.9425402, -1.2555897, 2.2726295, -3.3809295, 3.1981299

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9847399, upper bound: 2.9870103
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9841723, upper bound: 2.9864992
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.7334034, 1.6249501, -0.9439766, 1.8152119, -2.5486152, 2.5689259
1: -0.8890653, 1.5866382, -1.1733733, 1.7891084, -2.6781735, 2.7600117
2: -0.7032268, 1.8270780, -0.9287114, 2.1331313, -2.8363574, 2.7557893
3: -1.2971809, 1.6585422, -1.6920905, 1.8236557, -3.1208363, 3.3506327
4: -1.1083000, 1.9425402, -1.2756746, 2.3636315, -3.4719310, 3.2182148

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9847399, upper bound: 2.9870103
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9841723, upper bound: 2.9864992
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.9702232, 2.0391731, -0.9121016, 1.7768360, -2.7470589, 2.9512746
1: -1.1879681, 2.0189717, -1.1276571, 1.7428490, -2.9308171, 3.1466289
2: -0.9566915, 2.4168973, -0.8944148, 2.0478930, -3.0045846, 3.3113120
3: -1.7667999, 2.0601354, -1.6342735, 1.8015133, -3.5683129, 3.6944089
4: -1.3747511, 2.6204681, -1.2555897, 2.2726295, -3.6473806, 3.8760579

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9839280, upper bound: 2.9860957
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833035, upper bound: 2.9853391
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.9702232, 2.0391731, -0.9439766, 1.8152119, -2.7854350, 2.9831495
1: -1.1879681, 2.0189717, -1.1733733, 1.7891084, -2.9770765, 3.1923451
2: -0.9566915, 2.4168973, -0.9287114, 2.1331313, -3.0898223, 3.3456078
3: -1.7667999, 2.0601354, -1.6920905, 1.8236557, -3.5904555, 3.7522256
4: -1.3747511, 2.6204681, -1.2756746, 2.3636315, -3.7383826, 3.8961427

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9839280, upper bound: 2.9860957
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833035, upper bound: 2.9853391
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.9121016, 1.7768360, -0.7334034, 1.6249501, -2.5370512, 2.5102391
1: -1.1276571, 1.7428490, -0.8890653, 1.5866382, -2.7142954, 2.6319141
2: -0.8944148, 2.0478930, -0.7032268, 1.8270780, -2.7214928, 2.7511191
3: -1.6342735, 1.8015133, -1.2971809, 1.6585422, -3.2928150, 3.0986941
4: -1.2555897, 2.2726295, -1.1083000, 1.9425402, -3.1981299, 3.3809295

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9858470, upper bound: 2.9865514
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.9121016, 1.7768360, -0.9702232, 2.0391731, -2.9512742, 2.7470591
1: -1.1276571, 1.7428490, -1.1879681, 2.0189717, -3.1466289, 2.9308171
2: -0.8944148, 2.0478930, -0.9566915, 2.4168973, -3.3113120, 3.0045843
3: -1.6342735, 1.8015133, -1.7667999, 2.0601354, -3.6944084, 3.5683131
4: -1.2555897, 2.2726295, -1.3747511, 2.6204681, -3.8760579, 3.6473806

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9858470, upper bound: 2.9865514
time: 0.29 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.28 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.9439766, 1.8152119, -0.7334034, 1.6249501, -2.5689259, 2.5486152
1: -1.1733733, 1.7891084, -0.8890653, 1.5866382, -2.7600117, 2.6781733
2: -0.9287114, 2.1331313, -0.7032268, 1.8270780, -2.7557893, 2.8363571
3: -1.6920905, 1.8236557, -1.2971809, 1.6585422, -3.3506324, 3.1208363
4: -1.2756746, 2.3636315, -1.1083000, 1.9425402, -3.2182148, 3.4719315

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9855969, upper bound: 2.9852718
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.9439766, 1.8152119, -0.9702232, 2.0391731, -2.9831495, 2.7854345
1: -1.1733733, 1.7891084, -1.1879681, 2.0189717, -3.1923451, 2.9770765
2: -0.9287114, 2.1331313, -0.9566915, 2.4168973, -3.3456082, 3.0898223
3: -1.6920905, 1.8236557, -1.7667999, 2.0601354, -3.7522254, 3.5904555
4: -1.2756746, 2.3636315, -1.3747511, 2.6204681, -3.8961427, 3.7383826

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9855969, upper bound: 2.9852718
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
time: 0.28 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.9121016, 1.7768360, -0.9121016, 1.7768360, -2.6889372, 2.6889374
1: -1.1276571, 1.7428490, -1.1276571, 1.7428490, -2.8705058, 2.8705060
2: -0.8944148, 2.0478930, -0.8944148, 2.0478930, -2.9423077, 2.9423079
3: -1.6342735, 1.8015133, -1.6342735, 1.8015133, -3.4357867, 3.4357867
4: -1.2555897, 2.2726295, -1.2555897, 2.2726295, -3.5282192, 3.5282192

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9846274, upper bound: 2.9864402
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830620, upper bound: 2.9833834
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.9121016, 1.7768360, -0.9439766, 1.8152119, -2.7273128, 2.7208123
1: -1.1276571, 1.7428490, -1.1733733, 1.7891084, -2.9167655, 2.9162223
2: -0.8944148, 2.0478930, -0.9287114, 2.1331313, -3.0275459, 2.9766045
3: -1.6342735, 1.8015133, -1.6920905, 1.8236557, -3.4579291, 3.4936037
4: -1.2555897, 2.2726295, -1.2756746, 2.3636315, -3.6192212, 3.5483041

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9846274, upper bound: 2.9864402
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9829224, upper bound: 2.9833834
time: 0.30 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.9439766, 1.8152119, -0.9121016, 1.7768360, -2.7208123, 2.7273130
1: -1.1733733, 1.7891084, -1.1276571, 1.7428490, -2.9162223, 2.9167655
2: -0.9287114, 2.1331313, -0.8944148, 2.0478930, -2.9766040, 3.0275459
3: -1.6920905, 1.8236557, -1.6342735, 1.8015133, -3.4936037, 3.4579289
4: -1.2756746, 2.3636315, -1.2555897, 2.2726295, -3.5483041, 3.6192212

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9837944, upper bound: 2.9851426
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
time: 0.29 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.9439766, 1.8152119, -0.9439766, 1.8152119, -2.7591879, 2.7591877
1: -1.1733733, 1.7891084, -1.1733733, 1.7891084, -2.9624815, 2.9624815
2: -0.9287114, 2.1331313, -0.9287114, 2.1331313, -3.0618422, 3.0618420
3: -1.6920905, 1.8236557, -1.6920905, 1.8236557, -3.5157461, 3.5157461
4: -1.2756746, 2.3636315, -1.2756746, 2.3636315, -3.6393061, 3.6393061

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9837944, upper bound: 2.9851426
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
time: 0.30 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.20 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9875752, upper bound: 2.9871711
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9863687, upper bound: 2.9866600
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9875752, upper bound: 2.9871711
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9863687, upper bound: 2.9866600
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9860780, upper bound: 2.9862565
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9854999, upper bound: 2.9854999
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9860780, upper bound: 2.9862565
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9854999, upper bound: 2.9854999
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9847399, upper bound: 2.9870103
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9841723, upper bound: 2.9864992
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9847399, upper bound: 2.9870103
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9841723, upper bound: 2.9864992
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9839280, upper bound: 2.9860957
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9833035, upper bound: 2.9853391
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9839280, upper bound: 2.9860957
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9833035, upper bound: 2.9853391
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9858470, upper bound: 2.9865514
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9858470, upper bound: 2.9865514
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9855969, upper bound: 2.9852718
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9855969, upper bound: 2.9852718
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9846274, upper bound: 2.9864402
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9830620, upper bound: 2.9833834
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9846274, upper bound: 2.9864402
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9829224, upper bound: 2.9833834
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9837944, upper bound: 2.9851426
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9837944, upper bound: 2.9851426
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.20
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -0.7334034, 1.6249501, -2.3251481, 2.2897015
1: -0.8499212, 1.5115556, -0.8890653, 1.5866382, -2.4365594, 2.4006209
2: -0.6681871, 1.7435615, -0.7032268, 1.8270780, -2.4952650, 2.4467878
3: -1.2478759, 1.5815438, -1.2971809, 1.6585422, -2.9064181, 2.8787241
4: -1.0604532, 1.8642521, -1.1083000, 1.9425402, -3.0029933, 2.9725521

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9875287, upper bound: 2.9875287
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9875287, upper bound: 2.9875287
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -0.7334034, 1.6249501, -2.3369370, 2.2774985
1: -0.8672805, 1.5063487, -0.8890653, 1.5866382, -2.4539187, 2.3954139
2: -0.6787068, 1.7403395, -0.7032268, 1.8270780, -2.5057847, 2.4435658
3: -1.2585921, 1.5752970, -1.2971809, 1.6585422, -2.9171338, 2.8724773
4: -1.0591868, 1.8647437, -1.1083000, 1.9425402, -3.0017271, 2.9730437

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9875287, upper bound: 2.9875287
time: 0.28 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9875287, upper bound: 2.9875287
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -0.9702232, 2.0391731, -2.7393715, 2.5265214
1: -0.8499212, 1.5115556, -1.1879681, 2.0189717, -2.8688929, 2.6995237
2: -0.6681871, 1.7435615, -0.9566915, 2.4168973, -3.0850840, 2.7002525
3: -1.2478759, 1.5815438, -1.7667999, 2.0601354, -3.3080113, 3.3483434
4: -1.0604532, 1.8642521, -1.3747511, 2.6204681, -3.6809213, 3.2390032

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9863687, upper bound: 2.9866600
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9863687, upper bound: 2.9866600
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -0.9702232, 2.0391731, -2.7511599, 2.5143182
1: -0.8672805, 1.5063487, -1.1879681, 2.0189717, -2.8862522, 2.6943169
2: -0.6787068, 1.7403395, -0.9566915, 2.4168973, -3.0956035, 2.6970310
3: -1.2585921, 1.5752970, -1.7667999, 2.0601354, -3.3187275, 3.3420963
4: -1.0591868, 1.8647437, -1.3747511, 2.6204681, -3.6796548, 3.2394948

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9863687, upper bound: 2.9866600
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9863687, upper bound: 2.9866600
time: 0.28 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.9499983, 1.9835922, -0.7334034, 1.6249501, -2.5749483, 2.7169955
1: -1.1655170, 1.9660928, -0.8890653, 1.5866382, -2.7521553, 2.8551581
2: -0.9358571, 2.3591526, -0.7032268, 1.8270780, -2.7629352, 3.0623784
3: -1.7327125, 1.9976203, -1.2971809, 1.6585422, -3.3912547, 3.2948012
4: -1.3462847, 2.5602283, -1.1083000, 1.9425402, -3.2888250, 3.6685276

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9866600, upper bound: 2.9863687
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9866600, upper bound: 2.9863687
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.0122899, 2.0579705, -0.7334034, 1.6249501, -2.6372395, 2.7913735
1: -1.2475017, 2.0473447, -0.8890653, 1.5866382, -2.8341398, 2.9364100
2: -0.9972194, 2.4569695, -0.7032268, 1.8270780, -2.8242974, 3.1601958
3: -1.8210613, 2.0535314, -1.2971809, 1.6585422, -3.4796026, 3.3507123
4: -1.4084145, 2.6689005, -1.1083000, 1.9425402, -3.3509545, 3.7772005

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9866600, upper bound: 2.9863687
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9866600, upper bound: 2.9863687
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.9499983, 1.9835922, -0.9702232, 2.0391731, -2.9891715, 2.9538147
1: -1.1655170, 1.9660928, -1.1879681, 2.0189717, -3.1844888, 3.1540608
2: -0.9358571, 2.3591526, -0.9566915, 2.4168973, -3.3527532, 3.3158436
3: -1.7327125, 1.9976203, -1.7667999, 2.0601354, -3.7928476, 3.7644203
4: -1.3462847, 2.5602283, -1.3747511, 2.6204681, -3.9667530, 3.9349794

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854999, upper bound: 2.9854999
time: 0.27 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854999, upper bound: 2.9854999
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.0122899, 2.0579705, -0.9702232, 2.0391731, -3.0514627, 3.0281928
1: -1.2475017, 2.0473447, -1.1879681, 2.0189717, -3.2664733, 3.2353129
2: -0.9972194, 2.4569695, -0.9566915, 2.4168973, -3.4141164, 3.4136610
3: -1.8210613, 2.0535314, -1.7667999, 2.0601354, -3.8811960, 3.8203313
4: -1.4084145, 2.6689005, -1.3747511, 2.6204681, -4.0288825, 4.0436516

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854999, upper bound: 2.9854999
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854999, upper bound: 2.9854999
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -0.9121016, 1.7768360, -2.4770341, 2.4683998
1: -0.8499212, 1.5115556, -1.1276571, 1.7428490, -2.5927701, 2.6392126
2: -0.6681871, 1.7435615, -0.8944148, 2.0478930, -2.7160802, 2.6379757
3: -1.2478759, 1.5815438, -1.6342735, 1.8015133, -3.0493894, 3.2158172
4: -1.0604532, 1.8642521, -1.2555897, 2.2726295, -3.3330822, 3.1198418

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9844130, upper bound: 2.9866194
time: 0.28 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9844130, upper bound: 2.9866194
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -0.9121016, 1.7768360, -2.4888227, 2.4561968
1: -0.8672805, 1.5063487, -1.1276571, 1.7428490, -2.6101294, 2.6340058
2: -0.6787068, 1.7403395, -0.8944148, 2.0478930, -2.7265997, 2.6347542
3: -1.2585921, 1.5752970, -1.6342735, 1.8015133, -3.0601053, 3.2095702
4: -1.0591868, 1.8647437, -1.2555897, 2.2726295, -3.3318157, 3.1203334

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9844130, upper bound: 2.9866194
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9844130, upper bound: 2.9866194
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -0.9439766, 1.8152119, -2.5154102, 2.5002747
1: -0.8499212, 1.5115556, -1.1733733, 1.7891084, -2.6390295, 2.6849289
2: -0.6681871, 1.7435615, -0.9287114, 2.1331313, -2.8013182, 2.6722722
3: -1.2478759, 1.5815438, -1.6920905, 1.8236557, -3.0715318, 3.2736344
4: -1.0604532, 1.8642521, -1.2756746, 2.3636315, -3.4240847, 3.1399267

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9841723, upper bound: 2.9864824
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9841723, upper bound: 2.9864992
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -0.9439766, 1.8152119, -2.5271986, 2.4880714
1: -0.8672805, 1.5063487, -1.1733733, 1.7891084, -2.6563888, 2.6797221
2: -0.6787068, 1.7403395, -0.9287114, 2.1331313, -2.8118377, 2.6690509
3: -1.2585921, 1.5752970, -1.6920905, 1.8236557, -3.0822473, 3.2673874
4: -1.0591868, 1.8647437, -1.2756746, 2.3636315, -3.4228179, 3.1404183

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9841723, upper bound: 2.9864824
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9841723, upper bound: 2.9864992
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.9499983, 1.9835922, -0.9121016, 1.7768360, -2.7268343, 2.8956931
1: -1.1655170, 1.9660928, -1.1276571, 1.7428490, -2.9083657, 3.0937498
2: -0.9358571, 2.3591526, -0.8944148, 2.0478930, -2.9837501, 3.2535670
3: -1.7327125, 1.9976203, -1.6342735, 1.8015133, -3.5342259, 3.6318939
4: -1.3462847, 2.5602283, -1.2555897, 2.2726295, -3.6189141, 3.8158178

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835442, upper bound: 2.9854594
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835442, upper bound: 2.9854594
time: 0.29 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.0122899, 2.0579705, -0.9121016, 1.7768360, -2.7891254, 2.9700718
1: -1.2475017, 2.0473447, -1.1276571, 1.7428490, -2.9903505, 3.1750016
2: -0.9972194, 2.4569695, -0.8944148, 2.0478930, -3.0451126, 3.3513842
3: -1.8210613, 2.0535314, -1.6342735, 1.8015133, -3.6225746, 3.6878049
4: -1.4084145, 2.6689005, -1.2555897, 2.2726295, -3.6810441, 3.9244900

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835442, upper bound: 2.9854594
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835442, upper bound: 2.9854594
time: 0.28 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.9499983, 1.9835922, -0.9439766, 1.8152119, -2.7652102, 2.9275684
1: -1.1655170, 1.9660928, -1.1733733, 1.7891084, -2.9546249, 3.1394660
2: -0.9358571, 2.3591526, -0.9287114, 2.1331313, -3.0689878, 3.2878633
3: -1.7327125, 1.9976203, -1.6920905, 1.8236557, -3.5563684, 3.6897109
4: -1.3462847, 2.5602283, -1.2756746, 2.3636315, -3.7099161, 3.8359025

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9829629, upper bound: 2.9853223
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833035, upper bound: 2.9853391
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.0122899, 2.0579705, -0.9439766, 1.8152119, -2.8275013, 3.0019464
1: -1.2475017, 2.0473447, -1.1733733, 1.7891084, -3.0366099, 3.2207179
2: -0.9972194, 2.4569695, -0.9287114, 2.1331313, -3.1303506, 3.3856809
3: -1.8210613, 2.0535314, -1.6920905, 1.8236557, -3.6447170, 3.7456219
4: -1.4084145, 2.6689005, -1.2756746, 2.3636315, -3.7720461, 3.9445751

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833035, upper bound: 2.9853223
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833035, upper bound: 2.9853391
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.8800933, 1.7000839, -0.7334034, 1.6249501, -2.5050430, 2.4334874
1: -1.0882522, 1.6629176, -0.8890653, 1.5866382, -2.6748905, 2.5519829
2: -0.8591623, 1.9701490, -0.7032268, 1.8270780, -2.6862402, 2.6733754
3: -1.5843780, 1.7170482, -1.2971809, 1.6585422, -3.2429202, 3.0142286
4: -1.1997277, 2.1971321, -1.1083000, 1.9425402, -3.1422677, 3.3054321

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9866194, upper bound: 2.9844130
time: 0.29 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9866194, upper bound: 2.9844130
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.0258696, 1.8832996, -0.7334034, 1.6249501, -2.6508193, 2.6167030
1: -1.2680844, 1.8652637, -0.8890653, 1.5866382, -2.8547225, 2.7543290
2: -1.0059552, 2.2167563, -0.7032268, 1.8270780, -2.8330331, 2.9199822
3: -1.8244381, 1.9292028, -1.2971809, 1.6585422, -3.4829798, 3.2263832
4: -1.3557521, 2.4770975, -1.1083000, 1.9425402, -3.2982922, 3.5853975

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9866194, upper bound: 2.9844130
time: 0.29 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9866194, upper bound: 2.9844130
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.8800933, 1.7000839, -0.9702232, 2.0391731, -2.9192665, 2.6703069
1: -1.0882522, 1.6629176, -1.1879681, 2.0189717, -3.1072240, 2.8508859
2: -0.8591623, 1.9701490, -0.9566915, 2.4168973, -3.2760592, 2.9268405
3: -1.5843780, 1.7170482, -1.7667999, 2.0601354, -3.6445134, 3.4838476
4: -1.1997277, 2.1971321, -1.3747511, 2.6204681, -3.8201957, 3.5718832

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.29 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.0258696, 1.8832996, -0.9702232, 2.0391731, -3.0650427, 2.8535228
1: -1.2680844, 1.8652637, -1.1879681, 2.0189717, -3.2870560, 3.0532317
2: -1.0059552, 2.2167563, -0.9566915, 2.4168973, -3.4228513, 3.1734476
3: -1.8244381, 1.9292028, -1.7667999, 2.0601354, -3.8845735, 3.6960027
4: -1.3557521, 2.4770975, -1.3747511, 2.6204681, -3.9762201, 3.8518481

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.9274861, 1.7711896, -0.7334034, 1.6249501, -2.5524354, 2.5045929
1: -1.1543653, 1.7466712, -0.8890653, 1.5866382, -2.7410035, 2.6357365
2: -0.9117054, 2.0869017, -0.7032268, 1.8270780, -2.7387834, 2.7901282
3: -1.6646013, 1.7625735, -1.2971809, 1.6585422, -3.3231428, 3.0597544
4: -1.2472779, 2.3179801, -1.1083000, 1.9425402, -3.1898179, 3.4262800

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9864824, upper bound: 2.9841723
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9864824, upper bound: 2.9841723
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.0849965, 1.9832435, -0.7334034, 1.6249501, -2.7099459, 2.7166467
1: -1.3420913, 1.9703858, -0.8890653, 1.5866382, -2.9287295, 2.8594511
2: -1.0689102, 2.3526096, -0.7032268, 1.8270780, -2.8959880, 3.0558362
3: -1.9349499, 2.0002010, -1.2971809, 1.6585422, -3.5934918, 3.2973819
4: -1.4174156, 2.6281040, -1.1083000, 1.9425402, -3.3599558, 3.7364035

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9864992, upper bound: 2.9841723
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9864992, upper bound: 2.9841723
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.9274861, 1.7711896, -0.9702232, 2.0391731, -2.9666591, 2.7414126
1: -1.1543653, 1.7466712, -1.1879681, 2.0189717, -3.1733370, 2.9346390
2: -0.9117054, 2.0869017, -0.9566915, 2.4168973, -3.3286018, 3.0435932
3: -1.6646013, 1.7625735, -1.7667999, 2.0601354, -3.7247357, 3.5293734
4: -1.2472779, 2.3179801, -1.3747511, 2.6204681, -3.8677459, 3.6927311

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853223, upper bound: 2.9833035
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853223, upper bound: 2.9833035
time: 0.28 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.0849965, 1.9832435, -0.9702232, 2.0391731, -3.1241693, 2.9534662
1: -1.3420913, 1.9703858, -1.1879681, 2.0189717, -3.3610630, 3.1583538
2: -1.0689102, 2.3526096, -0.9566915, 2.4168973, -3.4858065, 3.3093011
3: -1.9349499, 2.0002010, -1.7667999, 2.0601354, -3.9950852, 3.7670009
4: -1.4174156, 2.6281040, -1.3747511, 2.6204681, -4.0378838, 4.0028553

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
time: 0.29 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
time: 0.29 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.8800933, 1.7000839, -0.9121016, 1.7768360, -2.6569293, 2.6121852
1: -1.0882522, 1.6629176, -1.1276571, 1.7428490, -2.8311012, 2.7905746
2: -0.8591623, 1.9701490, -0.8944148, 2.0478930, -2.9070554, 2.8645639
3: -1.5843780, 1.7170482, -1.6342735, 1.8015133, -3.3858914, 3.3513215
4: -1.1997277, 2.1971321, -1.2555897, 2.2726295, -3.4723573, 3.4527218

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835036, upper bound: 2.9835036
time: 0.29 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835036, upper bound: 2.9835036
time: 0.30 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.0258696, 1.8832996, -0.9121016, 1.7768360, -2.8027053, 2.7954011
1: -1.2680844, 1.8652637, -1.1276571, 1.7428490, -3.0109334, 2.9929206
2: -1.0059552, 2.2167563, -0.8944148, 2.0478930, -3.0538478, 3.1111710
3: -1.8244381, 1.9292028, -1.6342735, 1.8015133, -3.6259513, 3.5634761
4: -1.3557521, 2.4770975, -1.2555897, 2.2726295, -3.6283817, 3.7326865

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835036, upper bound: 2.9835036
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835036, upper bound: 2.9835036
time: 0.30 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.8800933, 1.7000839, -0.9439766, 1.8152119, -2.6953049, 2.6440601
1: -1.0882522, 1.6629176, -1.1733733, 1.7891084, -2.8773606, 2.8362908
2: -0.8591623, 1.9701490, -0.9287114, 2.1331313, -2.9922929, 2.8988602
3: -1.5843780, 1.7170482, -1.6920905, 1.8236557, -3.4080338, 3.4091382
4: -1.1997277, 2.1971321, -1.2756746, 2.3636315, -3.5633588, 3.4728067

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830620, upper bound: 2.9833666
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9829224, upper bound: 2.9833834
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.0258696, 1.8832996, -0.9439766, 1.8152119, -2.8410814, 2.8272762
1: -1.2680844, 1.8652637, -1.1733733, 1.7891084, -3.0571928, 3.0386369
2: -1.0059552, 2.2167563, -0.9287114, 2.1331313, -3.1390855, 3.1454678
3: -1.8244381, 1.9292028, -1.6920905, 1.8236557, -3.6480937, 3.6212933
4: -1.3557521, 2.4770975, -1.2756746, 2.3636315, -3.7193835, 3.7527716

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830620, upper bound: 2.9833666
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830620, upper bound: 2.9833834
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.9274861, 1.7711896, -0.9121016, 1.7768360, -2.7043216, 2.6832910
1: -1.1543653, 1.7466712, -1.1276571, 1.7428490, -2.8972144, 2.8743281
2: -0.9117054, 2.0869017, -0.8944148, 2.0478930, -2.9595983, 2.9813166
3: -1.6646013, 1.7625735, -1.6342735, 1.8015133, -3.4661138, 3.3968470
4: -1.2472779, 2.3179801, -1.2555897, 2.2726295, -3.5199075, 3.5735693

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833666, upper bound: 2.9832629
time: 0.29 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833666, upper bound: 2.9832629
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.0849965, 1.9832435, -0.9121016, 1.7768360, -2.8618321, 2.8953445
1: -1.3420913, 1.9703858, -1.1276571, 1.7428490, -3.0849404, 3.0980425
2: -1.0689102, 2.3526096, -0.8944148, 2.0478930, -3.1168032, 3.2470245
3: -1.9349499, 2.0002010, -1.6342735, 1.8015133, -3.7364631, 3.6344743
4: -1.4174156, 2.6281040, -1.2555897, 2.2726295, -3.6900451, 3.8836927

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833834, upper bound: 2.9832630
time: 0.29 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833834, upper bound: 2.9832630
time: 0.30 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.9274861, 1.7711896, -0.9439766, 1.8152119, -2.7426972, 2.7151661
1: -1.1543653, 1.7466712, -1.1733733, 1.7891084, -2.9434738, 2.9200444
2: -0.9117054, 2.0869017, -0.9287114, 2.1331313, -3.0448360, 3.0156131
3: -1.6646013, 1.7625735, -1.6920905, 1.8236557, -3.4882555, 3.4546640
4: -1.2472779, 2.3179801, -1.2756746, 2.3636315, -3.6109095, 3.5936546

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.0849965, 1.9832435, -0.9439766, 1.8152119, -2.9002080, 2.9272194
1: -1.3420913, 1.9703858, -1.1733733, 1.7891084, -3.1311998, 3.1437588
2: -1.0689102, 2.3526096, -0.9287114, 2.1331313, -3.2020409, 3.2813210
3: -1.9349499, 2.0002010, -1.6920905, 1.8236557, -3.7586055, 3.6922910
4: -1.4174156, 2.6281040, -1.2756746, 2.3636315, -3.7810471, 3.9037783

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
time: 0.31 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.33 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9875287, upper bound: 2.9875287
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9875287, upper bound: 2.9875287
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9875287, upper bound: 2.9875287
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9875287, upper bound: 2.9875287
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9863687, upper bound: 2.9866600
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9863687, upper bound: 2.9866600
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9863687, upper bound: 2.9866600
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9863687, upper bound: 2.9866600
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9866600, upper bound: 2.9863687
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9866600, upper bound: 2.9863687
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9866600, upper bound: 2.9863687
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9866600, upper bound: 2.9863687
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9854999, upper bound: 2.9854999
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9854999, upper bound: 2.9854999
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9854999, upper bound: 2.9854999
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9854999, upper bound: 2.9854999
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9844130, upper bound: 2.9866194
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9844130, upper bound: 2.9866194
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9844130, upper bound: 2.9866194
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9844130, upper bound: 2.9866194
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9841723, upper bound: 2.9864824
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9841723, upper bound: 2.9864992
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9841723, upper bound: 2.9864824
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9841723, upper bound: 2.9864992
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9835442, upper bound: 2.9854594
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9835442, upper bound: 2.9854594
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9835442, upper bound: 2.9854594
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9835442, upper bound: 2.9854594
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9829629, upper bound: 2.9853223
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9833035, upper bound: 2.9853391
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9833035, upper bound: 2.9853223
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9833035, upper bound: 2.9853391
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9866194, upper bound: 2.9844130
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9866194, upper bound: 2.9844130
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9866194, upper bound: 2.9844130
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9866194, upper bound: 2.9844130
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9864824, upper bound: 2.9841723
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9864824, upper bound: 2.9841723
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9864992, upper bound: 2.9841723
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9864992, upper bound: 2.9841723
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9853223, upper bound: 2.9833035
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9853223, upper bound: 2.9833035
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9835036, upper bound: 2.9835036
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9835036, upper bound: 2.9835036
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9835036, upper bound: 2.9835036
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9835036, upper bound: 2.9835036
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9830620, upper bound: 2.9833666
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9829224, upper bound: 2.9833834
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9830620, upper bound: 2.9833666
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9830620, upper bound: 2.9833834
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9833666, upper bound: 2.9832629
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9833666, upper bound: 2.9832629
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9833834, upper bound: 2.9832630
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9833834, upper bound: 2.9832630
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.33
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831427

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -0.7001984, 1.5562983, -2.2564964, 2.2564967
1: -0.8499212, 1.5115556, -0.8499212, 1.5115556, -2.3614769, 2.3614769
2: -0.6681871, 1.7435615, -0.6681871, 1.7435615, -2.4117484, 2.4117484
3: -1.2478759, 1.5815438, -1.2478759, 1.5815438, -2.8294196, 2.8294196
4: -1.0604532, 1.8642521, -1.0604532, 1.8642521, -2.9247050, 2.9247050

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 43

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9867412, upper bound: 2.9872799
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9887810, upper bound: 2.9873026
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -0.7119868, 1.5440953, -2.2442937, 2.2682850
1: -0.8499212, 1.5115556, -0.8672805, 1.5063487, -2.3562698, 2.3788362
2: -0.6681871, 1.7435615, -0.6787068, 1.7403395, -2.4085267, 2.4222682
3: -1.2478759, 1.5815438, -1.2585921, 1.5752970, -2.8231730, 2.8401353
4: -1.0604532, 1.8642521, -1.0591868, 1.8647437, -2.9251969, 2.9234388

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 43

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9867412, upper bound: 2.9872799
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9887810, upper bound: 2.9873026
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -0.7001984, 1.5562983, -2.2682850, 2.2442937
1: -0.8672805, 1.5063487, -0.8499212, 1.5115556, -2.3788362, 2.3562698
2: -0.6787068, 1.7403395, -0.6681871, 1.7435615, -2.4222684, 2.4085267
3: -1.2585921, 1.5752970, -1.2478759, 1.5815438, -2.8401353, 2.8231728
4: -1.0591868, 1.8647437, -1.0604532, 1.8642521, -2.9234390, 2.9251969

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9847572, upper bound: 2.9851154
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9845003, upper bound: 2.9845869
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -0.7119868, 1.5440953, -2.2560821, 2.2560821
1: -0.8672805, 1.5063487, -0.8672805, 1.5063487, -2.3736291, 2.3736291
2: -0.6787068, 1.7403395, -0.6787068, 1.7403395, -2.4190464, 2.4190459
3: -1.2585921, 1.5752970, -1.2585921, 1.5752970, -2.8338890, 2.8338888
4: -1.0591868, 1.8647437, -1.0591868, 1.8647437, -2.9239306, 2.9239306

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9847572, upper bound: 2.9851154
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9845003, upper bound: 2.9850021
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -0.9499983, 1.9835922, -2.6837907, 2.5062966
1: -0.8499212, 1.5115556, -1.1655170, 1.9660928, -2.8160141, 2.6770723
2: -0.6681871, 1.7435615, -0.9358571, 2.3591526, -3.0273392, 2.6794181
3: -1.2478759, 1.5815438, -1.7327125, 1.9976203, -3.2454963, 3.3142562
4: -1.0604532, 1.8642521, -1.3462847, 2.5602283, -3.6206815, 3.2105370

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 43

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9858867, upper bound: 2.9861935
time: 0.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9839130, upper bound: 2.9839139
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -1.0122899, 2.0579705, -2.7581689, 2.5685880
1: -0.8499212, 1.5115556, -1.2475017, 2.0473447, -2.8972659, 2.7590573
2: -0.6681871, 1.7435615, -0.9972194, 2.4569695, -3.1251566, 2.7407808
3: -1.2478759, 1.5815438, -1.8210613, 2.0535314, -3.3014073, 3.4026046
4: -1.0604532, 1.8642521, -1.4084145, 2.6689005, -3.7293537, 3.2726665

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 43

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9858867, upper bound: 2.9862453
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9839130, upper bound: 2.9839139
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -0.9499983, 1.9835922, -2.6955791, 2.4940934
1: -0.8672805, 1.5063487, -1.1655170, 1.9660928, -2.8333731, 2.6718657
2: -0.6787068, 1.7403395, -0.9358571, 2.3591526, -3.0378590, 2.6761966
3: -1.2585921, 1.5752970, -1.7327125, 1.9976203, -3.2562125, 3.3080089
4: -1.0591868, 1.8647437, -1.3462847, 2.5602283, -3.6194148, 3.2110286

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9847094, upper bound: 2.9850684
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9844524, upper bound: 2.9845399
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -1.0122899, 2.0579705, -2.7699573, 2.5563850
1: -0.8672805, 1.5063487, -1.2475017, 2.0473447, -2.9146252, 2.7538505
2: -0.6787068, 1.7403395, -0.9972194, 2.4569695, -3.1356764, 2.7375586
3: -1.2585921, 1.5752970, -1.8210613, 2.0535314, -3.3121235, 3.3963578
4: -1.0591868, 1.8647437, -1.4084145, 2.6689005, -3.7280872, 3.2731581

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9847094, upper bound: 2.9850684
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9844524, upper bound: 2.9845399
time: 0.29 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.9499983, 1.9835922, -0.7001984, 1.5562983, -2.5062966, 2.6837907
1: -1.1655170, 1.9660928, -0.8499212, 1.5115556, -2.6770725, 2.8160141
2: -0.9358571, 2.3591526, -0.6681871, 1.7435615, -2.6794178, 3.0273397
3: -1.7327125, 1.9976203, -1.2478759, 1.5815438, -3.3142560, 3.2454963
4: -1.3462847, 2.5602283, -1.0604532, 1.8642521, -3.2105367, 3.6206808

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9862358, upper bound: 2.9863880
time: 0.33 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9870152, upper bound: 2.9863880
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.9499983, 1.9835922, -0.7119868, 1.5440953, -2.4940934, 2.6955791
1: -1.1655170, 1.9660928, -0.8672805, 1.5063487, -2.6718655, 2.8333733
2: -0.9358571, 2.3591526, -0.6787068, 1.7403395, -2.6761966, 3.0378590
3: -1.7327125, 1.9976203, -1.2585921, 1.5752970, -3.3080096, 3.2562122
4: -1.3462847, 2.5602283, -1.0591868, 1.8647437, -3.2110286, 3.6194148

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9862358, upper bound: 2.9863880
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9870152, upper bound: 2.9863880
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.0122899, 2.0579705, -0.7001984, 1.5562983, -2.5685883, 2.7581689
1: -1.2475017, 2.0473447, -0.8499212, 1.5115556, -2.7590573, 2.8972659
2: -0.9972194, 2.4569695, -0.6681871, 1.7435615, -2.7407808, 3.1251566
3: -1.8210613, 2.0535314, -1.2478759, 1.5815438, -3.4026043, 3.3014073
4: -1.4084145, 2.6689005, -1.0604532, 1.8642521, -3.2726665, 3.7293534

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9864098, upper bound: 2.9856314
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9839118, upper bound: 2.9838973
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9840397, upper bound: 2.9839488
time: 0.32 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.0122899, 2.0579705, -0.7119868, 1.5440953, -2.5563850, 2.7699573
1: -1.2475017, 2.0473447, -0.8672805, 1.5063487, -2.7538505, 2.9146252
2: -0.9972194, 2.4569695, -0.6787068, 1.7403395, -2.7375588, 3.1356764
3: -1.8210613, 2.0535314, -1.2585921, 1.5752970, -3.3963575, 3.3121235
4: -1.4084145, 2.6689005, -1.0591868, 1.8647437, -3.2731581, 3.7280874

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9864099, upper bound: 2.9856314
time: 0.31 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9839118, upper bound: 2.9838973
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9840397, upper bound: 2.9843076
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.9499983, 1.9835922, -0.9499983, 1.9835922, -2.9335904, 2.9335904
1: -1.1655170, 1.9660928, -1.1655170, 1.9660928, -3.1316099, 3.1316099
2: -0.9358571, 2.3591526, -0.9358571, 2.3591526, -3.2950087, 3.2950089
3: -1.7327125, 1.9976203, -1.7327125, 1.9976203, -3.7303329, 3.7303329
4: -1.3462847, 2.5602283, -1.3462847, 2.5602283, -3.9065132, 3.9065132

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9845601, upper bound: 2.9849144
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838660, upper bound: 2.9838733
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.9499983, 1.9835922, -1.0122899, 2.0579705, -3.0079689, 2.9958818
1: -1.1655170, 1.9660928, -1.2475017, 2.0473447, -3.2128615, 3.2135944
2: -0.9358571, 2.3591526, -0.9972194, 2.4569695, -3.3928266, 3.3563716
3: -1.7327125, 1.9976203, -1.8210613, 2.0535314, -3.7862439, 3.8186817
4: -1.3462847, 2.5602283, -1.4084145, 2.6689005, -4.0151854, 3.9686427

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9845601, upper bound: 2.9849144
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838660, upper bound: 2.9838733
time: 0.31 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.0122899, 2.0579705, -0.9499983, 1.9835922, -2.9958820, 3.0079689
1: -1.2475017, 2.0473447, -1.1655170, 1.9660928, -3.2135944, 3.2128615
2: -0.9972194, 2.4569695, -0.9358571, 2.3591526, -3.3563716, 3.3928266
3: -1.8210613, 2.0535314, -1.7327125, 1.9976203, -3.8186817, 3.7862439
4: -1.4084145, 2.6689005, -1.3462847, 2.5602283, -3.9686427, 4.0151854

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838639, upper bound: 2.9838503
time: 0.32 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9839918, upper bound: 2.9839018
time: 0.30 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.0122899, 2.0579705, -1.0122899, 2.0579705, -3.0702600, 3.0702600
1: -1.2475017, 2.0473447, -1.2475017, 2.0473447, -3.2948465, 3.2948465
2: -0.9972194, 2.4569695, -0.9972194, 2.4569695, -3.4541888, 3.4541888
3: -1.8210613, 2.0535314, -1.8210613, 2.0535314, -3.8745928, 3.8745928
4: -1.4084145, 2.6689005, -1.4084145, 2.6689005, -4.0773149, 4.0773149

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838639, upper bound: 2.9838503
time: 0.30 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9839918, upper bound: 2.9839018
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -0.8800933, 1.7000839, -2.4002824, 2.4363916
1: -0.8499212, 1.5115556, -1.0882522, 1.6629176, -2.5128388, 2.5998077
2: -0.6681871, 1.7435615, -0.8591623, 1.9701490, -2.6383362, 2.6027234
3: -1.2478759, 1.5815438, -1.5843780, 1.7170482, -2.9649239, 3.1659217
4: -1.0604532, 1.8642521, -1.1997277, 2.1971321, -3.2575853, 3.0639796

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 43

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9841678, upper bound: 2.9860167
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9825486, upper bound: 2.9838390
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -1.0258696, 1.8832996, -2.5834980, 2.5821679
1: -0.8499212, 1.5115556, -1.2680844, 1.8652637, -2.7151849, 2.7796400
2: -0.6681871, 1.7435615, -1.0059552, 2.2167563, -2.8849428, 2.7495162
3: -1.2478759, 1.5815438, -1.8244381, 1.9292028, -3.1770787, 3.4059815
4: -1.0604532, 1.8642521, -1.3557521, 2.4770975, -3.5375497, 3.2200038

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 43

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9841678, upper bound: 2.9860583
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9825486, upper bound: 2.9839107
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -0.8800933, 1.7000839, -2.4120708, 2.4241886
1: -0.8672805, 1.5063487, -1.0882522, 1.6629176, -2.5301981, 2.5946009
2: -0.6787068, 1.7403395, -0.8591623, 1.9701490, -2.6488557, 2.5995016
3: -1.2585921, 1.5752970, -1.5843780, 1.7170482, -2.9756398, 3.1596751
4: -1.0591868, 1.8647437, -1.1997277, 2.1971321, -3.2563190, 3.0644712

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833449, upper bound: 2.9849934
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830879, upper bound: 2.9844649
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -1.0258696, 1.8832996, -2.5952864, 2.5699649
1: -0.8672805, 1.5063487, -1.2680844, 1.8652637, -2.7325442, 2.7744331
2: -0.6787068, 1.7403395, -1.0059552, 2.2167563, -2.8954630, 2.7462947
3: -1.2585921, 1.5752970, -1.8244381, 1.9292028, -3.1877947, 3.3997345
4: -1.0591868, 1.8647437, -1.3557521, 2.4770975, -3.5362837, 3.2204957

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833449, upper bound: 2.9849934
time: 0.29 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830879, upper bound: 2.9845054
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -0.9274861, 1.7711896, -2.4713879, 2.4837842
1: -0.8499212, 1.5115556, -1.1543653, 1.7466712, -2.5965922, 2.6659207
2: -0.6681871, 1.7435615, -0.9117054, 2.0869017, -2.7550888, 2.6552663
3: -1.2478759, 1.5815438, -1.6646013, 1.7625735, -3.0104494, 3.2461443
4: -1.0604532, 1.8642521, -1.2472779, 2.3179801, -3.3784328, 3.1115298

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 43

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838820, upper bound: 2.9859007
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9824287, upper bound: 2.9837292
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.7001984, 1.5562983, -1.0849965, 1.9832435, -2.6834416, 2.6412945
1: -0.8499212, 1.5115556, -1.3420913, 1.9703858, -2.8203070, 2.8536468
2: -0.6681871, 1.7435615, -1.0689102, 2.3526096, -3.0207968, 2.8124709
3: -1.2478759, 1.5815438, -1.9349499, 2.0002010, -3.2480769, 3.5164936
4: -1.0604532, 1.8642521, -1.4174156, 2.6281040, -3.6885567, 3.2816677

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 16
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 43

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838820, upper bound: 2.9860145
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9824287, upper bound: 2.9838668
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -0.9274861, 1.7711896, -2.4831762, 2.4715810
1: -0.8672805, 1.5063487, -1.1543653, 1.7466712, -2.6139517, 2.6607141
2: -0.6787068, 1.7403395, -0.9117054, 2.0869017, -2.7656083, 2.6520448
3: -1.2585921, 1.5752970, -1.6646013, 1.7625735, -3.0211656, 3.2398975
4: -1.0591868, 1.8647437, -1.2472779, 2.3179801, -3.3771667, 3.1120214

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9832250, upper bound: 2.9848836
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9829681, upper bound: 2.9843551
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.7119868, 1.5440953, -1.0849965, 1.9832435, -2.6952302, 2.6290917
1: -0.8672805, 1.5063487, -1.3420913, 1.9703858, -2.8376663, 2.8484402
2: -0.6787068, 1.7403395, -1.0689102, 2.3526096, -3.0313163, 2.8092494
3: -1.2585921, 1.5752970, -1.9349499, 2.0002010, -3.2587929, 3.5102465
4: -1.0591868, 1.8647437, -1.4174156, 2.6281040, -3.6872902, 3.2821593

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9832250, upper bound: 2.9848836
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9829681, upper bound: 2.9844362
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.9499983, 1.9835922, -0.8800933, 1.7000839, -2.6500821, 2.8636851
1: -1.1655170, 1.9660928, -1.0882522, 1.6629176, -2.8284345, 3.0543449
2: -0.9358571, 2.3591526, -0.8591623, 1.9701490, -2.9060061, 3.2183142
3: -1.7327125, 1.9976203, -1.5843780, 1.7170482, -3.4497604, 3.5819983
4: -1.3462847, 2.5602283, -1.1997277, 2.1971321, -3.5434170, 3.7599559

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9832159, upper bound: 2.9848394
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9825015, upper bound: 2.9837983
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.9499983, 1.9835922, -1.0258696, 1.8832996, -2.8332977, 3.0094614
1: -1.1655170, 1.9660928, -1.2680844, 1.8652637, -3.0307808, 3.2341769
2: -0.9358571, 2.3591526, -1.0059552, 2.2167563, -3.1526134, 3.3651071
3: -1.7327125, 1.9976203, -1.8244381, 1.9292028, -3.6619148, 3.8220584
4: -1.3462847, 2.5602283, -1.3557521, 2.4770975, -3.8233814, 3.9159803

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9832159, upper bound: 2.9849112
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9825015, upper bound: 2.9838701
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.0122899, 2.0579705, -0.8800933, 1.7000839, -2.7123737, 2.9380639
1: -1.2475017, 2.0473447, -1.0882522, 1.6629176, -2.9104190, 3.1355968
2: -0.9972194, 2.4569695, -0.8591623, 1.9701490, -2.9673686, 3.3161318
3: -1.8210613, 2.0535314, -1.5843780, 1.7170482, -3.5381088, 3.6379094
4: -1.4084145, 2.6689005, -1.1997277, 2.1971321, -3.6055465, 3.8686280

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9824995, upper bound: 2.9837753
time: 0.30 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9826273, upper bound: 2.9838269
time: 0.32 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.0122899, 2.0579705, -1.0258696, 1.8832996, -2.8955894, 3.0838394
1: -1.2475017, 2.0473447, -1.2680844, 1.8652637, -3.1127653, 3.3154292
2: -0.9972194, 2.4569695, -1.0059552, 2.2167563, -3.2139757, 3.4629242
3: -1.8210613, 2.0535314, -1.8244381, 1.9292028, -3.7502637, 3.8779695
4: -1.4084145, 2.6689005, -1.3557521, 2.4770975, -3.8855109, 4.0246525

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9824995, upper bound: 2.9837753
time: 0.31 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9826273, upper bound: 2.9838588
time: 0.30 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.9499983, 1.9835922, -0.9274861, 1.7711896, -2.7211878, 2.9110782
1: -1.1655170, 1.9660928, -1.1543653, 1.7466712, -2.9121881, 3.1204581
2: -0.9358571, 2.3591526, -0.9117054, 2.0869017, -3.0227585, 3.2708573
3: -1.7327125, 1.9976203, -1.6646013, 1.7625735, -3.4952860, 3.6622217
4: -1.3462847, 2.5602283, -1.2472779, 2.3179801, -3.6642647, 3.8075061

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830352, upper bound: 2.9847296
time: 0.32 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9823817, upper bound: 2.9836885
time: 0.33 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.9499983, 1.9835922, -1.0849965, 1.9832435, -2.9332416, 3.0685883
1: -1.1655170, 1.9660928, -1.3420913, 1.9703858, -3.1359026, 3.3081841
2: -0.9358571, 2.3591526, -1.0689102, 2.3526096, -3.2884667, 3.4280629
3: -1.7327125, 1.9976203, -1.9349499, 2.0002010, -3.7329133, 3.9325702
4: -1.3462847, 2.5602283, -1.4174156, 2.6281040, -3.9743886, 3.9776440

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 16

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830352, upper bound: 2.9848673
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9823817, upper bound: 2.9838262
time: 0.31 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.0122899, 2.0579705, -0.9274861, 1.7711896, -2.7834795, 2.9854565
1: -1.2475017, 2.0473447, -1.1543653, 1.7466712, -2.9941728, 3.2017100
2: -0.9972194, 2.4569695, -0.9117054, 2.0869017, -3.0841212, 3.3686748
3: -1.8210613, 2.0535314, -1.6646013, 1.7625735, -3.5836349, 3.7181323
4: -1.4084145, 2.6689005, -1.2472779, 2.3179801, -3.7263947, 3.9161782

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9820011, upper bound: 2.9836655
time: 0.35 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9825074, upper bound: 2.9837171
time: 0.34 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.0122899, 2.0579705, -1.0849965, 1.9832435, -2.9955330, 3.1429663
1: -1.2475017, 2.0473447, -1.3420913, 1.9703858, -3.2178874, 3.3894360
2: -0.9972194, 2.4569695, -1.0689102, 2.3526096, -3.3498292, 3.5258799
3: -1.8210613, 2.0535314, -1.9349499, 2.0002010, -3.8212619, 3.9884810
4: -1.4084145, 2.6689005, -1.4174156, 2.6281040, -4.0365186, 4.0863161

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 16

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9823796, upper bound: 2.9836655
time: 0.33 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9825075, upper bound: 2.9837863
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.8800933, 1.7000839, -0.7001984, 1.5562983, -2.4363916, 2.4002824
1: -1.0882522, 1.6629176, -0.8499212, 1.5115556, -2.5998077, 2.5128388
2: -0.8591623, 1.9701490, -0.6681871, 1.7435615, -2.6027236, 2.6383362
3: -1.5843780, 1.7170482, -1.2478759, 1.5815438, -3.1659217, 2.9649241
4: -1.1997277, 2.1971321, -1.0604532, 1.8642521, -3.0639796, 3.2575848

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9865550, upper bound: 2.9859520
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9847160, upper bound: 2.9855277
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838672, upper bound: 2.9831014
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.8800933, 1.7000839, -0.7119868, 1.5440953, -2.4241886, 2.4120708
1: -1.0882522, 1.6629176, -0.8672805, 1.5063487, -2.5946009, 2.5301981
2: -0.8591623, 1.9701490, -0.6787068, 1.7403395, -2.5995018, 2.6488557
3: -1.5843780, 1.7170482, -1.2585921, 1.5752970, -3.1596746, 2.9756396
4: -1.1997277, 2.1971321, -1.0591868, 1.8647437, -3.0644712, 3.2563188

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9865550, upper bound: 2.9859520
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9847160, upper bound: 2.9855277
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838672, upper bound: 2.9831014
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.0258696, 1.8832996, -0.7001984, 1.5562983, -2.5821679, 2.5834980
1: -1.2680844, 1.8652637, -0.8499212, 1.5115556, -2.7796400, 2.7151847
2: -1.0059552, 2.2167563, -0.6681871, 1.7435615, -2.7495165, 2.8849435
3: -1.8244381, 1.9292028, -1.2478759, 1.5815438, -3.4059808, 3.1770787
4: -1.3557521, 2.4770975, -1.0604532, 1.8642521, -3.2200041, 3.5375507

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 45

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9863728, upper bound: 2.9836757
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9866194, upper bound: 2.9844130
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9866194, upper bound: 2.9844130
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.0258696, 1.8832996, -0.7119868, 1.5440953, -2.5699646, 2.5952864
1: -1.2680844, 1.8652637, -0.8672805, 1.5063487, -2.7744331, 2.7325439
2: -1.0059552, 2.2167563, -0.6787068, 1.7403395, -2.7462945, 2.8954630
3: -1.8244381, 1.9292028, -1.2585921, 1.5752970, -3.3997345, 3.1877944
4: -1.3557521, 2.4770975, -1.0591868, 1.8647437, -3.2204957, 3.5362835

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 45

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9863728, upper bound: 2.9836757
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9866194, upper bound: 2.9844130
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9866194, upper bound: 2.9844130
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.8800933, 1.7000839, -0.9499983, 1.9835922, -2.8636856, 2.6500821
1: -1.0882522, 1.6629176, -1.1655170, 1.9660928, -3.0543449, 2.8284345
2: -0.8591623, 1.9701490, -0.9358571, 2.3591526, -3.2183142, 2.9060059
3: -1.5843780, 1.7170482, -1.7327125, 1.9976203, -3.5819983, 3.4497604
4: -1.1997277, 2.1971321, -1.3462847, 2.5602283, -3.7599556, 3.5434170

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9844186, upper bound: 2.9854293
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838194, upper bound: 2.9830498
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.8800933, 1.7000839, -1.0122899, 2.0579705, -2.9380636, 2.7123737
1: -1.0882522, 1.6629176, -1.2475017, 2.0473447, -3.1355968, 2.9104195
2: -0.8591623, 1.9701490, -0.9972194, 2.4569695, -3.3161318, 2.9673686
3: -1.5843780, 1.7170482, -1.8210613, 2.0535314, -3.6379094, 3.5381091
4: -1.1997277, 2.1971321, -1.4084145, 2.6689005, -3.8686278, 3.6055465

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9844186, upper bound: 2.9855550
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9838194, upper bound: 2.9831706
time: 0.30 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.0258696, 1.8832996, -0.9499983, 1.9835922, -3.0094614, 2.8332977
1: -1.2680844, 1.8652637, -1.1655170, 1.9660928, -3.2341771, 3.0307808
2: -1.0059552, 2.2167563, -0.9358571, 2.3591526, -3.3651066, 3.1526129
3: -1.8244381, 1.9292028, -1.7327125, 1.9976203, -3.8220584, 3.6619148
4: -1.3557521, 2.4770975, -1.3462847, 2.5602283, -3.9159803, 3.8233819

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 45

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.0258696, 1.8832996, -1.0122899, 2.0579705, -3.0838394, 2.8955894
1: -1.2680844, 1.8652637, -1.2475017, 2.0473447, -3.3154292, 3.1127653
2: -1.0059552, 2.2167563, -0.9972194, 2.4569695, -3.4629247, 3.2139757
3: -1.8244381, 1.9292028, -1.8210613, 2.0535314, -3.8779695, 3.7502639
4: -1.3557521, 2.4770975, -1.4084145, 2.6689005, -4.0246525, 3.8855119

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 45

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9854594, upper bound: 2.9835442
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.9274861, 1.7711896, -0.7001984, 1.5562983, -2.4837842, 2.4713879
1: -1.1543653, 1.7466712, -0.8499212, 1.5115556, -2.6659207, 2.5965924
2: -0.9117054, 2.0869017, -0.6681871, 1.7435615, -2.6552665, 2.7550886
3: -1.6646013, 1.7625735, -1.2478759, 1.5815438, -3.2461436, 3.0104494
4: -1.2472779, 2.3179801, -1.0604532, 1.8642521, -3.1115298, 3.3784332

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 43

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9862358, upper bound: 2.9850348
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9842173, upper bound: 2.9840632
time: 0.30 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9837447, upper bound: 2.9830813
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.9274861, 1.7711896, -0.7119868, 1.5440953, -2.4715810, 2.4831762
1: -1.1543653, 1.7466712, -0.8672805, 1.5063487, -2.6607141, 2.6139517
2: -0.9117054, 2.0869017, -0.6787068, 1.7403395, -2.6520448, 2.7656083
3: -1.6646013, 1.7625735, -1.2585921, 1.5752970, -3.2398968, 3.0211656
4: -1.2472779, 2.3179801, -1.0591868, 1.8647437, -3.1120214, 3.3771667

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 43

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9862358, upper bound: 2.9850348
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9842173, upper bound: 2.9844199
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9837447, upper bound: 2.9835747
time: 0.33 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.0849965, 1.9832435, -0.7001984, 1.5562983, -2.6412942, 2.6834419
1: -1.3420913, 1.9703858, -0.8499212, 1.5115556, -2.8536468, 2.8203070
2: -1.0689102, 2.3526096, -0.6681871, 1.7435615, -2.8124709, 3.0207968
3: -1.9349499, 2.0002010, -1.2478759, 1.5815438, -3.5164936, 3.2480769
4: -1.4174156, 2.6281040, -1.0604532, 1.8642521, -3.2816677, 3.6885567

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9863177, upper bound: 2.9834350
time: 0.31 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9864992, upper bound: 2.9841723
time: 0.34 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9864992, upper bound: 2.9841723
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.0849965, 1.9832435, -0.7119868, 1.5440953, -2.6290915, 2.6952302
1: -1.3420913, 1.9703858, -0.8672805, 1.5063487, -2.8484402, 2.8376663
2: -1.0689102, 2.3526096, -0.6787068, 1.7403395, -2.8092492, 3.0313163
3: -1.9349499, 2.0002010, -1.2585921, 1.5752970, -3.5102463, 3.2587931
4: -1.4174156, 2.6281040, -1.0591868, 1.8647437, -3.2821593, 3.6872902

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9863177, upper bound: 2.9834350
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9864992, upper bound: 2.9841723
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9864992, upper bound: 2.9841723
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.9274861, 1.7711896, -0.9499983, 1.9835922, -2.9110780, 2.7211878
1: -1.1543653, 1.7466712, -1.1655170, 1.9660928, -3.1204581, 2.9121881
2: -0.9117054, 2.0869017, -0.9358571, 2.3591526, -3.2708578, 3.0227587
3: -1.6646013, 1.7625735, -1.7327125, 1.9976203, -3.6622217, 3.4952860
4: -1.2472779, 2.3179801, -1.3462847, 2.5602283, -3.8075061, 3.6642647

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 43

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9840834, upper bound: 2.9839687
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9836969, upper bound: 2.9830295
time: 0.31 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.9274861, 1.7711896, -1.0122899, 2.0579705, -2.9854562, 2.7834792
1: -1.1543653, 1.7466712, -1.2475017, 2.0473447, -3.2017100, 2.9941728
2: -0.9117054, 2.0869017, -0.9972194, 2.4569695, -3.3686748, 3.0841212
3: -1.6646013, 1.7625735, -1.8210613, 2.0535314, -3.7181327, 3.5836349
4: -1.2472779, 2.3179801, -1.4084145, 2.6689005, -3.9161782, 3.7263947

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 43

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9840834, upper bound: 2.9840942
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9836969, upper bound: 2.9830295
time: 0.32 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.0849965, 1.9832435, -0.9499983, 1.9835922, -3.0685885, 2.9332418
1: -1.3420913, 1.9703858, -1.1655170, 1.9660928, -3.3081834, 3.1359026
2: -1.0689102, 2.3526096, -0.9358571, 2.3591526, -3.4280615, 3.2884667
3: -1.9349499, 2.0002010, -1.7327125, 1.9976203, -3.9325702, 3.7329130
4: -1.4174156, 2.6281040, -1.3462847, 2.5602283, -3.9776440, 3.9743886

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 35

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853223, upper bound: 2.9833035
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.0849965, 1.9832435, -1.0122899, 2.0579705, -3.1429667, 2.9955330
1: -1.3420913, 1.9703858, -1.2475017, 2.0473447, -3.3894360, 3.2178872
2: -1.0689102, 2.3526096, -0.9972194, 2.4569695, -3.5258799, 3.3498292
3: -1.9349499, 2.0002010, -1.8210613, 2.0535314, -3.9884813, 3.8212621
4: -1.4174156, 2.6281040, -1.4084145, 2.6689005, -4.0863161, 4.0365181

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 35

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
time: 0.33 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9853391, upper bound: 2.9833035
time: 0.32 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.8800933, 1.7000839, -0.8800933, 1.7000839, -2.5801773, 2.5801773
1: -1.0882522, 1.6629176, -1.0882522, 1.6629176, -2.7511697, 2.7511697
2: -0.8591623, 1.9701490, -0.8591623, 1.9701490, -2.8293114, 2.8293114
3: -1.5843780, 1.7170482, -1.5843780, 1.7170482, -3.3014257, 3.3014257
4: -1.1997277, 2.1971321, -1.1997277, 2.1971321, -3.3968596, 3.3968596

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9840454, upper bound: 2.9854058
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9824549, upper bound: 2.9829795
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.8800933, 1.7000839, -1.0258696, 1.8832996, -2.7633929, 2.7259536
1: -1.0882522, 1.6629176, -1.2680844, 1.8652637, -2.9535160, 2.9310021
2: -0.8591623, 1.9701490, -1.0059552, 2.2167563, -3.0759184, 2.9761043
3: -1.5843780, 1.7170482, -1.8244381, 1.9292028, -3.5135808, 3.5414858
4: -1.1997277, 2.1971321, -1.3557521, 2.4770975, -3.6768241, 3.5528841

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9840454, upper bound: 2.9854775
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9824549, upper bound: 2.9830513
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.0258696, 1.8832996, -0.8800933, 1.7000839, -2.7259536, 2.7633929
1: -1.2680844, 1.8652637, -1.0882522, 1.6629176, -2.9310021, 2.9535158
2: -1.0059552, 2.2167563, -0.8591623, 1.9701490, -2.9761043, 3.0759180
3: -1.8244381, 1.9292028, -1.5843780, 1.7170482, -3.5414858, 3.5135803
4: -1.3557521, 2.4770975, -1.1997277, 2.1971321, -3.5528841, 3.6768250

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 45

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835036, upper bound: 2.9835036
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.0258696, 1.8832996, -1.0258696, 1.8832996, -2.9091692, 2.9091692
1: -1.2680844, 1.8652637, -1.2680844, 1.8652637, -3.1333480, 3.1333480
2: -1.0059552, 2.2167563, -1.0059552, 2.2167563, -3.2227111, 3.2227108
3: -1.8244381, 1.9292028, -1.8244381, 1.9292028, -3.7536407, 3.7536407
4: -1.3557521, 2.4770975, -1.3557521, 2.4770975, -3.8328490, 3.8328490

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 45

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835036, upper bound: 2.9835036
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.8800933, 1.7000839, -0.9274861, 1.7711896, -2.6512828, 2.6275697
1: -1.0882522, 1.6629176, -1.1543653, 1.7466712, -2.8349233, 2.8172829
2: -0.8591623, 1.9701490, -0.9117054, 2.0869017, -2.9460638, 2.8818541
3: -1.5843780, 1.7170482, -1.6646013, 1.7625735, -3.3469515, 3.3816488
4: -1.1997277, 2.1971321, -1.2472779, 2.3179801, -3.5177078, 3.4444098

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835110, upper bound: 2.9852960
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9819565, upper bound: 2.9828697
time: 0.31 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.8800933, 1.7000839, -1.0849965, 1.9832435, -2.8633368, 2.7850804
1: -1.0882522, 1.6629176, -1.3420913, 1.9703858, -3.0586379, 3.0050089
2: -0.8591623, 1.9701490, -1.0689102, 2.3526096, -3.2117720, 3.0390592
3: -1.5843780, 1.7170482, -1.9349499, 2.0002010, -3.5845790, 3.6519976
4: -1.1997277, 2.1971321, -1.4174156, 2.6281040, -3.8278308, 3.6145477

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9835110, upper bound: 2.9854337
time: 0.31 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9819565, upper bound: 2.9828697
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.0258696, 1.8832996, -0.9274861, 1.7711896, -2.7970591, 2.8107853
1: -1.2680844, 1.8652637, -1.1543653, 1.7466712, -3.0147555, 3.0196290
2: -1.0059552, 2.2167563, -0.9117054, 2.0869017, -3.0928564, 3.1284614
3: -1.8244381, 1.9292028, -1.6646013, 1.7625735, -3.5870116, 3.5938039
4: -1.3557521, 2.4770975, -1.2472779, 2.3179801, -3.6737323, 3.7243752

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 45

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9829224, upper bound: 2.9833666
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.0258696, 1.8832996, -1.0849965, 1.9832435, -3.0091128, 2.9682961
1: -1.2680844, 1.8652637, -1.3420913, 1.9703858, -3.2384701, 3.2073550
2: -1.0059552, 2.2167563, -1.0689102, 2.3526096, -3.3585646, 3.2856665
3: -1.8244381, 1.9292028, -1.9349499, 2.0002010, -3.8246386, 3.8641524
4: -1.3557521, 2.4770975, -1.4174156, 2.6281040, -3.9838552, 3.8945122

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 35
type: A, layer: 3, pos: 45

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830620, upper bound: 2.9833666
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.9274861, 1.7711896, -0.8800933, 1.7000839, -2.6275697, 2.6512828
1: -1.1543653, 1.7466712, -1.0882522, 1.6629176, -2.8172829, 2.8349233
2: -0.9117054, 2.0869017, -0.8591623, 1.9701490, -2.8818541, 2.9460640
3: -1.6646013, 1.7625735, -1.5843780, 1.7170482, -3.3816485, 3.3469515
4: -1.2472779, 2.3179801, -1.1997277, 2.1971321, -3.4444098, 3.5177078

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 43

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9831771, upper bound: 2.9839413
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9823324, upper bound: 2.9829594
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.9274861, 1.7711896, -1.0258696, 1.8832996, -2.8107855, 2.7970591
1: -1.1543653, 1.7466712, -1.2680844, 1.8652637, -3.0196290, 3.0147552
2: -0.9117054, 2.0869017, -1.0059552, 2.2167563, -3.1284618, 3.0928566
3: -1.6646013, 1.7625735, -1.8244381, 1.9292028, -3.5938039, 3.5870113
4: -1.2472779, 2.3179801, -1.3557521, 2.4770975, -3.7243752, 3.6737320

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 43

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9831771, upper bound: 2.9840130
time: 0.30 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9823324, upper bound: 2.9830311
time: 0.32 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.0849965, 1.9832435, -0.8800933, 1.7000839, -2.7850804, 2.8633366
1: -1.3420913, 1.9703858, -1.0882522, 1.6629176, -3.0050089, 3.0586381
2: -1.0689102, 2.3526096, -0.8591623, 1.9701490, -3.0390592, 3.2117720
3: -1.9349499, 2.0002010, -1.5843780, 1.7170482, -3.6519971, 3.5845790
4: -1.4174156, 2.6281040, -1.1997277, 2.1971321, -3.6145477, 3.8278313

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 35

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833834, upper bound: 2.9832630
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.0849965, 1.9832435, -1.0258696, 1.8832996, -2.9682961, 3.0091131
1: -1.3420913, 1.9703858, -1.2680844, 1.8652637, -3.2073550, 3.2384701
2: -1.0689102, 2.3526096, -1.0059552, 2.2167563, -3.2856665, 3.3585646
3: -1.9349499, 2.0002010, -1.8244381, 1.9292028, -3.8641527, 3.8246388
4: -1.4174156, 2.6281040, -1.3557521, 2.4770975, -3.8945127, 3.9838560

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 35

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9833834, upper bound: 2.9832630
time: 0.33 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.9274861, 1.7711896, -0.9274861, 1.7711896, -2.6986752, 2.6986752
1: -1.1543653, 1.7466712, -1.1543653, 1.7466712, -2.9010365, 2.9010365
2: -0.9117054, 2.0869017, -0.9117054, 2.0869017, -2.9986072, 2.9986072
3: -1.6646013, 1.7625735, -1.6646013, 1.7625735, -3.4271741, 3.4271741
4: -1.2472779, 2.3179801, -1.2472779, 2.3179801, -3.5652580, 3.5652580

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 43

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9829596, upper bound: 2.9838315
time: 0.32 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821813, upper bound: 2.9828496
time: 0.33 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.9274861, 1.7711896, -1.0849965, 1.9832435, -2.9107289, 2.8561859
1: -1.1543653, 1.7466712, -1.3420913, 1.9703858, -3.1247511, 3.0887625
2: -0.9117054, 2.0869017, -1.0689102, 2.3526096, -3.2643151, 3.1558118
3: -1.6646013, 1.7625735, -1.9349499, 2.0002010, -3.6648016, 3.6975234
4: -1.2472779, 2.3179801, -1.4174156, 2.6281040, -3.8753817, 3.7353954

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 43

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9829596, upper bound: 2.9839691
time: 0.34 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9821813, upper bound: 2.9829873
time: 0.34 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.0849965, 1.9832435, -0.9274861, 1.7711896, -2.8561859, 2.9107289
1: -1.3420913, 1.9703858, -1.1543653, 1.7466712, -3.0887625, 3.1247511
2: -1.0689102, 2.3526096, -0.9117054, 2.0869017, -3.1558118, 3.2643151
3: -1.9349499, 2.0002010, -1.6646013, 1.7625735, -3.6975234, 3.6648021
4: -1.4174156, 2.6281040, -1.2472779, 2.3179801, -3.7353952, 3.8753815

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 35

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259
time: 0.35 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.0849965, 1.9832435, -1.0849965, 1.9832435, -3.0682392, 3.0682392
1: -1.3420913, 1.9703858, -1.3420913, 1.9703858, -3.3124771, 3.3124769
2: -1.0689102, 2.3526096, -1.0689102, 2.3526096, -3.4215198, 3.4215198
3: -1.9349499, 2.0002010, -1.9349499, 2.0002010, -3.9351504, 3.9351509
4: -1.4174156, 2.6281040, -1.4174156, 2.6281040, -4.0455189, 4.0455184

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 41
type: A, layer: 3, pos: 32
type: A, layer: 3, pos: 0
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 35

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 41

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 32

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 0

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 45

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9830196, upper bound: 2.9831259
time: 0.36 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.51 + 386.42 = 388.93 seconds
