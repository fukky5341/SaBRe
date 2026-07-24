## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 6)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.566640958


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2967296, 1.2967298)
1: (-2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4835277, 1.4835272)
2: (1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2275105, 1.2275107)
3: (-6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0011253, 1.0011251)
4: (-2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9452736, 0.9452736)
5: (-4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0601525, 1.0601530)
6: (-4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4067559, 1.4067559)
7: (-8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8539648, 0.8539647)
8: (-4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4132328, 1.4132328)
9: (-11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9780154, 0.9780157)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.90 + 36.67 = 59.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.5694884, upper bound: 0.5694897

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4625
type: DSZ, layer: 1, pos: 4632
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4625

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5693726, upper bound: 0.5665205
time: 5.92 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665226, upper bound: 0.5693720
time: 5.27 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 11.44 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 11.44
Output dim: 2, lower bound: -0.5693726, upper bound: 0.5665205
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 11.44
Output dim: 2, lower bound: -0.5665226, upper bound: 0.5693720

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2979150, 1.2963262
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4813247, 1.4893470
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2328267, 1.2254872
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0056090, 0.9994314
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9468502, 0.9446752
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0587420, 1.0638645
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4047771, 1.4119682
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8535533, 0.8551276
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4128642, 1.4141941
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9790168, 0.9776344

Time for backsubstitution: 21.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4632
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 4632

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5693709, upper bound: 0.5654419
time: 5.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5682918, upper bound: 0.5665210
time: 5.30 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2963262, 1.2967298
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4835277, 1.4813247
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2254872, 1.2275107
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9994311, 1.0011251
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9446751, 0.9452736
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0601525, 1.0587420
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4067559, 1.4047771
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8539648, 0.8535532
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4132328, 1.4128642
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9776344, 0.9780157

Time for backsubstitution: 22.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4632
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 4632

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5665196, upper bound: 0.5682915
time: 4.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5654404, upper bound: 0.5693706
time: 4.84 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 32.59 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 32.59
Output dim: 2, lower bound: -0.5693709, upper bound: 0.5654419
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 32.59
Output dim: 2, lower bound: -0.5682918, upper bound: 0.5665210
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 32.59
Output dim: 2, lower bound: -0.5665196, upper bound: 0.5682915
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 32.59
Output dim: 2, lower bound: -0.5654404, upper bound: 0.5693706

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2978940, 1.2963023
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4799452, 1.4882441
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2321706, 1.2247007
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0057507, 0.9995425
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9464974, 0.9443812
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0588634, 1.0640206
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4047000, 1.4119215
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8510129, 0.8530097
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4124446, 1.4136899
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9762921, 0.9753628

Time for backsubstitution: 22.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5654989, upper bound: 0.5653724
time: 4.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5693042, upper bound: 0.5615590
time: 5.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2978911, 1.2963052
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4802213, 1.4879680
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2320399, 1.2248316
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0057206, 0.9995728
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9465561, 0.9443226
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0588982, 1.0639858
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4047306, 1.4118910
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8514352, 0.8525873
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4123597, 1.4137743
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9767451, 0.9749098

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5644197, upper bound: 0.5664516
time: 4.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5682250, upper bound: 0.5626380
time: 6.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2963052, 1.2967057
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4821486, 1.4802213
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2248316, 1.2267246
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9995728, 1.0012367
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9443226, 0.9449804
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0602744, 1.0588982
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4066789, 1.4047303
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8514242, 0.8514352
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4128127, 1.4123604
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9749098, 0.9757433

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5626388, upper bound: 0.5682237
time: 4.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5664528, upper bound: 0.5644189
time: 4.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2963018, 1.2967088
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4824247, 1.4799452
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2247005, 1.2268555
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9995427, 1.0012670
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9443812, 0.9449217
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0603087, 1.0588636
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4067094, 1.4046998
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8518465, 0.8510129
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4127283, 1.4124444
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9753628, 0.9752903

Time for backsubstitution: 22.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5615597, upper bound: 0.5693032
time: 4.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5653737, upper bound: 0.5654977
time: 5.89 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 32.58 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 32.58
Output dim: 2, lower bound: -0.5654989, upper bound: 0.5653724
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 2, lower bound: -0.5693042, upper bound: 0.5615590
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 32.58
Output dim: 2, lower bound: -0.5644197, upper bound: 0.5664516
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 2, lower bound: -0.5682250, upper bound: 0.5626380
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 2, lower bound: -0.5626388, upper bound: 0.5682237
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 32.58
Output dim: 2, lower bound: -0.5664528, upper bound: 0.5644189
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 2, lower bound: -0.5615597, upper bound: 0.5693032
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 32.58
Output dim: 2, lower bound: -0.5653737, upper bound: 0.5654977

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2979002, 1.2963071
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4799495, 1.4882474
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2321639, 1.2246923
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0057645, 0.9995604
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9464709, 0.9443498
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0588820, 1.0640433
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4046884, 1.4119120
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8510170, 0.8530129
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4124250, 1.4136732
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9762862, 0.9753554

Time for backsubstitution: 22.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 904

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5652310, upper bound: 0.5615591
time: 4.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5693022, upper bound: 0.5574973
time: 4.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2978969, 1.2963102
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4802256, 1.4879713
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2320328, 1.2248230
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0057340, 0.9995904
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9465295, 0.9442911
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0589168, 1.0640087
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4047189, 1.4118814
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8514392, 0.8525906
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4123411, 1.4137576
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9767392, 0.9749024

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 904

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5641524, upper bound: 0.5626365
time: 4.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5682230, upper bound: 0.5585778
time: 4.39 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2963099, 1.2967124
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4821510, 1.4802256
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2248230, 1.2267172
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9995904, 1.0012500
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9442912, 0.9449539
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0602968, 1.0589168
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4066691, 1.4047189
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8514280, 0.8514392
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4127965, 1.4123409
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9749024, 0.9757373

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 904

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5585768, upper bound: 0.5682224
time: 4.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5626368, upper bound: 0.5641531
time: 4.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2963071, 1.2967153
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4824271, 1.4799495
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2246919, 1.2268481
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9995604, 1.0012803
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9443499, 0.9448953
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0603316, 1.0588822
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.4066997, 1.4046884
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8518503, 0.8510170
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4127121, 1.4124248
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9753554, 0.9752843

Time for backsubstitution: 22.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 904
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 904

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5574982, upper bound: 0.5693014
time: 4.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5615576, upper bound: 0.5652303
time: 4.62 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 31.75 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.75
Output dim: 2, lower bound: -0.5652310, upper bound: 0.5615591
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 2, lower bound: -0.5693022, upper bound: 0.5574973
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 31.75
Output dim: 2, lower bound: -0.5641524, upper bound: 0.5626365
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 2, lower bound: -0.5682230, upper bound: 0.5585778
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 2, lower bound: -0.5585768, upper bound: 0.5682224
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.75
Output dim: 2, lower bound: -0.5626368, upper bound: 0.5641531
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 2, lower bound: -0.5574982, upper bound: 0.5693014
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 31.75
Output dim: 2, lower bound: -0.5615576, upper bound: 0.5652303

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2906799, 1.2902927
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4642200, 1.4693704
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2267303, 1.2181726
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0067244, 1.0007675
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9415514, 0.9386134
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0671632, 1.0743759
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3889751, 1.3988156
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8590863, 0.8593386
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4111662, 1.4125931
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9742203, 0.9729459

Time for backsubstitution: 23.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.36 seconds

### Candidate
type: DSZ, layer: 1, pos: 143

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5689380, upper bound: 0.5574943
time: 5.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5692993, upper bound: 0.5571372
time: 5.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2906766, 1.2902956
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4644961, 1.4690943
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2265997, 1.2183034
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0066938, 1.0007977
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9416101, 0.9385548
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0671980, 1.0743411
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3890057, 1.3987851
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8595088, 0.8589163
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4110823, 1.4126775
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9746733, 0.9724927

Time for backsubstitution: 22.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 1, pos: 143

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5678588, upper bound: 0.5585729
time: 4.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5682201, upper bound: 0.5582162
time: 4.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2902956, 1.2894912
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4632745, 1.4644961
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2183027, 1.2212834
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0007982, 1.0022097
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9385545, 0.9400342
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0706303, 1.0671980
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3935728, 1.3890052
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8577535, 0.8595085
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4117165, 1.4110820
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9724927, 0.9736717

Time for backsubstitution: 22.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 143

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5582157, upper bound: 0.5682186
time: 6.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5585739, upper bound: 0.5678603
time: 4.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2902927, 1.2894940
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4635506, 1.4642200
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2181721, 1.2214143
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -1.0007677, 1.0022399
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9386132, 0.9399755
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0706646, 1.0671632
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3936033, 1.3889747
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8581758, 0.8590863
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.4116325, 1.4111660
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9729462, 0.9732187

Time for backsubstitution: 22.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 143
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 143

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5571383, upper bound: 0.5692981
time: 4.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5574953, upper bound: 0.5689389
time: 4.20 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 31.63 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 31.63
Output dim: 2, lower bound: -0.5689380, upper bound: 0.5574943
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 31.63
Output dim: 2, lower bound: -0.5692993, upper bound: 0.5571372
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 31.63
Output dim: 2, lower bound: -0.5678588, upper bound: 0.5585729
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 31.63
Output dim: 2, lower bound: -0.5682201, upper bound: 0.5582162
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 31.63
Output dim: 2, lower bound: -0.5582157, upper bound: 0.5682186
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 31.63
Output dim: 2, lower bound: -0.5585739, upper bound: 0.5678603
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 31.63
Output dim: 2, lower bound: -0.5571383, upper bound: 0.5692981
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 31.63
Output dim: 2, lower bound: -0.5574953, upper bound: 0.5689389

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2916946, 1.2903814
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4667134, 1.4712882
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2266893, 1.2185562
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9910018, 0.9875734
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9412724, 0.9382768
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0637102, 1.0715077
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3827620, 1.3934617
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8564591, 0.8561817
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.3957014, 1.3992887
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9736600, 0.9722717

Time for backsubstitution: 22.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 927

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5642803, upper bound: 0.5574915
time: 4.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5689355, upper bound: 0.5528457
time: 4.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2907686, 1.2913074
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4661384, 1.4718633
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2271142, 1.2181315
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9935296, 0.9850454
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9412149, 0.9383340
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0642948, 1.0709231
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3836207, 1.3926029
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8559294, 0.8567115
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.3978615, 1.3971286
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9735460, 0.9723854

Time for backsubstitution: 22.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 927

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 927

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.5646417, upper bound: 0.5571349
time: 4.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.5692965, upper bound: 0.5524870
time: 4.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -10.4847822, -8.6198349, -10.4847822, -8.6198349, -1.2916918, 1.2903845
1: -2.9937477, -1.2634897, -2.9937477, -1.2634897, -1.4669895, 1.4710121
2: 1.9920239, 3.3824191, 1.9920239, 3.3824191, -1.2265587, 1.2186871
3: -6.9481716, -5.5183735, -6.9481716, -5.5183735, -0.9909713, 0.9876034
4: -2.0612500, -0.6644926, -2.0612500, -0.6644926, -0.9413310, 0.9382181
5: -4.3456483, -3.0169828, -4.3456483, -3.0169828, -1.0637455, 1.0714729
6: -4.3337641, -2.5347354, -4.3337641, -2.5347354, -1.3827925, 1.3934312
7: -8.5212250, -7.2025671, -8.5212250, -7.2025671, -0.8568814, 0.8557594
8: -4.3148742, -2.7288890, -4.3148742, -2.7288890, -1.3956170, 1.3993731
9: -11.8572426, -10.1520758, -11.8572426, -10.1520758, -0.9741130, 0.9718184

Time for backsubstitution: 22.66 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 59.57 + 553.30 = 612.87 seconds
