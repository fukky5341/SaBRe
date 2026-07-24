## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_5.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 317.9962633056


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-147.6918945, 296.2000732, -147.6918945, 296.2000732, -443.8919678, 443.8919678)
1: (-54.2157822, 114.7664490, -54.2157822, 114.7664490, -168.9822083, 168.9822083)
2: (-26.4642086, 118.5023804, -26.4642086, 118.5023804, -144.9665833, 144.9665833)
3: (-64.3750992, 135.0477753, -64.3750992, 135.0477753, -199.4228668, 199.4228668)
4: (-33.7146263, 118.1404877, -33.7146263, 118.1404877, -151.8550873, 151.8550873)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.22 + 1.73 = 3.94 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -320.5607493, upper bound: 320.5607493

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5514484, upper bound: 320.5564180
time: 0.59 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5509673, upper bound: 320.5509673
time: 0.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.38 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 0, lower bound: -320.5514484, upper bound: 320.5564180
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.38
Output dim: 0, lower bound: -320.5509673, upper bound: 320.5509673

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -131.0137939, 268.4029541, -147.6918945, 296.2000732, -427.2138672, 416.0947876
1: -48.8644409, 102.7274017, -54.2157822, 114.7664490, -163.6308594, 156.9431763
2: -23.5231190, 107.4005661, -26.4642086, 118.5023804, -142.0254974, 133.8647308
3: -57.5010719, 121.1244354, -64.3750992, 135.0477753, -192.5488434, 185.4995117
4: -29.8939457, 106.9961319, -33.7146263, 118.1404877, -148.0343933, 140.7107544

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5509673, upper bound: 320.5509673
time: 0.63 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5509673, upper bound: 320.5509673
time: 0.63 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -124.7892914, 263.5599060, -137.5716400, 279.3168335, -404.1061401, 401.1315308
1: -47.5410271, 99.7803802, -50.9776382, 107.2982788, -154.8393097, 150.7579956
2: -22.4581146, 105.4499359, -24.6796627, 111.7233887, -134.1814880, 130.1295929
3: -55.5659828, 117.2088928, -60.2208443, 126.5813065, -182.1472931, 177.4297333
4: -28.5510330, 104.8793106, -31.4015141, 111.3261032, -139.8770905, 136.2808228

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5509673, upper bound: 320.5509673
time: 0.60 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5509673, upper bound: 320.5509673
time: 0.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.87 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.87
Output dim: 0, lower bound: -320.5509673, upper bound: 320.5509673
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.87
Output dim: 0, lower bound: -320.5509673, upper bound: 320.5509673
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.87
Output dim: 0, lower bound: -320.5509673, upper bound: 320.5509673
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.87
Output dim: 0, lower bound: -320.5509673, upper bound: 320.5509673

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -131.0137939, 268.4029541, -131.0137939, 268.4029541, -399.4167480, 399.4167480
1: -48.8644409, 102.7274017, -48.8644409, 102.7274017, -151.5918427, 151.5918427
2: -23.5231190, 107.4005661, -23.5231190, 107.4005661, -130.9236603, 130.9236603
3: -57.5010719, 121.1244354, -57.5010719, 121.1244354, -178.6254883, 178.6254883
4: -29.8939457, 106.9961319, -29.8939457, 106.9961319, -136.8900452, 136.8900604

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5448929, upper bound: 320.5212235
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5489374, upper bound: 320.5543632
time: 0.69 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -131.0137939, 268.4029541, -124.7892914, 263.5599060, -394.5737000, 393.1922302
1: -48.8644409, 102.7274017, -47.5410271, 99.7803802, -148.6448212, 150.2684326
2: -23.5231190, 107.4005661, -22.4581146, 105.4499359, -128.9730530, 129.8586731
3: -57.5010719, 121.1244354, -55.5659828, 117.2088928, -174.7099609, 176.6903992
4: -29.8939457, 106.9961319, -28.5510330, 104.8793106, -134.7732544, 135.5471649

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5448929, upper bound: 320.5212235
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5489374, upper bound: 320.5543632
time: 0.60 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -124.7892914, 263.5599060, -131.0137939, 268.4029541, -393.1922302, 394.5737000
1: -47.5410271, 99.7803802, -48.8644409, 102.7274017, -150.2684326, 148.6448212
2: -22.4581146, 105.4499359, -23.5231190, 107.4005661, -129.8586731, 128.9730530
3: -55.5659828, 117.2088928, -57.5010719, 121.1244354, -176.6903992, 174.7099609
4: -28.5510330, 104.8793106, -29.8939457, 106.9961319, -135.5471649, 134.7732544

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5462952, upper bound: 320.5415589
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5509673, upper bound: 320.5509673
time: 0.64 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -124.7892914, 263.5599060, -124.7892914, 263.5599060, -388.3491821, 388.3491821
1: -47.5410271, 99.7803802, -47.5410271, 99.7803802, -147.3213959, 147.3213959
2: -22.4581146, 105.4499359, -22.4581146, 105.4499359, -127.9080505, 127.9080505
3: -55.5659828, 117.2088928, -55.5659828, 117.2088928, -172.7748718, 172.7748718
4: -28.5510330, 104.8793106, -28.5510330, 104.8793106, -133.4303436, 133.4303436

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5462952, upper bound: 320.5415589
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5509673, upper bound: 320.5509673
time: 0.59 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.43 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -320.5448929, upper bound: 320.5212235
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -320.5489374, upper bound: 320.5543632
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -320.5448929, upper bound: 320.5212235
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -320.5489374, upper bound: 320.5543632
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -320.5462952, upper bound: 320.5415589
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -320.5509673, upper bound: 320.5509673
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -320.5462952, upper bound: 320.5415589
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.43
Output dim: 0, lower bound: -320.5509673, upper bound: 320.5509673

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -119.1956711, 250.1245117, -126.6343536, 261.3728333, -380.5685120, 376.7588501
1: -45.3708763, 95.1940155, -47.5199661, 99.8680191, -145.2388916, 142.7139893
2: -21.5172729, 100.2001801, -22.7625198, 104.6347885, -126.1520462, 122.9626923
3: -53.1180153, 112.0457077, -55.8008003, 117.7006226, -170.8186340, 167.8465118
4: -27.3980484, 99.6261292, -28.9376335, 104.1806030, -131.5786438, 128.5637512

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5426211, upper bound: 320.5426211
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5426211, upper bound: 320.5438376
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -116.2315903, 244.8721466, -127.2160339, 262.3126221, -378.5441589, 372.0881958
1: -44.3751526, 93.1720428, -47.7072296, 100.2435074, -144.6186371, 140.8792725
2: -20.9795532, 98.1318283, -22.8679581, 104.9934769, -125.9730225, 120.9997864
3: -51.9186134, 109.6377792, -56.0377998, 118.1536407, -170.0722504, 165.6755524
4: -26.7571697, 97.5490417, -29.0678024, 104.5485229, -131.3056946, 126.6168289

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5438376, upper bound: 320.5540346
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5438376, upper bound: 320.5552520
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -119.1956711, 250.1245117, -120.0807190, 256.0333862, -375.2290649, 370.2052307
1: -45.3708763, 95.1940155, -46.1741562, 96.8061066, -142.1769714, 141.3681641
2: -21.5172729, 100.2001801, -21.6506290, 102.5281754, -124.0454407, 121.8507996
3: -53.1180153, 112.0457077, -53.8974190, 113.6145401, -166.7325592, 165.9431305
4: -27.3980484, 99.6261292, -27.5766544, 101.8758850, -129.2739258, 127.2027740

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5347244, upper bound: 320.4911539
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5437256, upper bound: 320.5187503
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -116.2315903, 244.8721466, -121.4606094, 258.1633301, -374.3948975, 366.3327026
1: -44.3751526, 93.1720428, -46.5661583, 97.6511917, -142.0263062, 139.7382050
2: -20.9795532, 98.1318283, -21.8806591, 103.3513107, -124.3308487, 120.0124893
3: -51.9186134, 109.6377792, -54.3737450, 114.6292038, -166.5478210, 164.0115204
4: -26.7571697, 97.5490417, -27.8532581, 102.7206573, -129.4778290, 125.4022903

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5409405, upper bound: 320.5478957
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5489374, upper bound: 320.5543632
time: 0.58 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -139.4374542, 283.2442322, -127.7415237, 261.3328247, -400.7702637, 410.9857483
1: -51.3943214, 107.9217072, -47.6509399, 100.1896591, -151.5839539, 155.5726471
2: -24.9015865, 113.2208786, -22.9715462, 104.5628357, -129.4644165, 136.1924286
3: -60.6053391, 126.9218979, -56.1874962, 118.1781464, -178.7834778, 183.1093903
4: -31.4384499, 112.8451767, -29.2250538, 104.1792068, -135.6176300, 142.0702209

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5488248, upper bound: 320.5418866
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5488248, upper bound: 320.5418866
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -123.1938248, 260.6712341, -131.0137939, 268.4029541, -391.5967407, 391.6850281
1: -47.0105515, 98.5678024, -48.8644409, 102.7274017, -149.7379456, 147.4322357
2: -22.1759129, 104.2996216, -23.5231190, 107.4005661, -129.5764771, 127.8227386
3: -54.9006386, 115.8525772, -57.5010719, 121.1244354, -176.0250244, 173.3536072
4: -28.1930389, 103.7236404, -29.8939457, 106.9961319, -135.1891785, 133.6175690

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5561412, upper bound: 320.5490345
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5561412, upper bound: 320.5514484
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -139.4374542, 283.2442322, -121.9947357, 257.7613525, -397.1987915, 405.2389526
1: -51.3943214, 107.9217072, -46.4731598, 97.6722183, -149.0665436, 154.3948669
2: -24.9015865, 113.2208786, -21.9806709, 103.1917725, -128.0933533, 135.2015533
3: -60.6053391, 126.9218979, -54.4337044, 114.7146301, -175.3199615, 181.3556061
4: -31.4384499, 112.8451767, -27.9623890, 102.5478745, -133.9863281, 140.8075714

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5413530, upper bound: 320.5413530
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5413530, upper bound: 320.5415589
time: 3.19 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -123.1938248, 260.6712341, -124.7892914, 263.5599060, -386.7537231, 385.4605103
1: -47.0105515, 98.5678024, -47.5410271, 99.7803802, -146.7909241, 146.1088104
2: -22.1759129, 104.2996216, -22.4581146, 105.4499359, -127.6258469, 126.7577362
3: -54.9006386, 115.8525772, -55.5659828, 117.2088928, -172.1095276, 171.4185181
4: -28.1930389, 103.7236404, -28.5510330, 104.8793106, -133.0723419, 132.2746735

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5415589, upper bound: 320.5462952
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5415589, upper bound: 320.5509673
time: 0.62 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.50 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5426211, upper bound: 320.5426211
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5426211, upper bound: 320.5438376
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5438376, upper bound: 320.5540346
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5438376, upper bound: 320.5552520
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5347244, upper bound: 320.4911539
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5437256, upper bound: 320.5187503
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5409405, upper bound: 320.5478957
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5489374, upper bound: 320.5543632
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5488248, upper bound: 320.5418866
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5488248, upper bound: 320.5418866
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5561412, upper bound: 320.5490345
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5561412, upper bound: 320.5514484
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5413530, upper bound: 320.5413530
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5413530, upper bound: 320.5415589
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5415589, upper bound: 320.5462952
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.50
Output dim: 0, lower bound: -320.5415589, upper bound: 320.5509673

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -119.1956711, 250.1245117, -119.1956711, 250.1245117, -369.3201904, 369.3201904
1: -45.3708763, 95.1940155, -45.3708763, 95.1940155, -140.5648956, 140.5648956
2: -21.5172729, 100.2001801, -21.5172729, 100.2001801, -121.7174377, 121.7174377
3: -53.1180153, 112.0457077, -53.1180153, 112.0457077, -165.1637268, 165.1637268
4: -27.3980484, 99.6261292, -27.3980484, 99.6261292, -127.0241699, 127.0241699

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5312137, upper bound: 320.5415570
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5301496, upper bound: 320.5301496
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -119.1956711, 250.1245117, -116.2315903, 244.8721466, -364.0678101, 366.3560486
1: -45.3708763, 95.1940155, -44.3751526, 93.1720428, -138.5429230, 139.5691681
2: -21.5172729, 100.2001801, -20.9795532, 98.1318283, -119.6491013, 121.1797256
3: -53.1180153, 112.0457077, -51.9186134, 109.6377792, -162.7557678, 163.9643250
4: -27.3980484, 99.6261292, -26.7571697, 97.5490417, -124.9470825, 126.3833008

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5312137, upper bound: 320.5438376
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5301496, upper bound: 320.5324303
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -116.2315903, 244.8721466, -119.1956711, 250.1245117, -366.3560486, 364.0678101
1: -44.3751526, 93.1720428, -45.3708763, 95.1940155, -139.5691528, 138.5429230
2: -20.9795532, 98.1318283, -21.5172729, 100.2001801, -121.1797256, 119.6491013
3: -51.9186134, 109.6377792, -53.1180153, 112.0457077, -163.9643250, 162.7557831
4: -26.7571697, 97.5490417, -27.3980484, 99.6261292, -126.3833008, 124.9470825

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5529707
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5529707
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -116.2315903, 244.8721466, -116.2315903, 244.8721466, -361.1036682, 361.1036377
1: -44.3751526, 93.1720428, -44.3751526, 93.1720428, -137.5471802, 137.5471802
2: -20.9795532, 98.1318283, -20.9795532, 98.1318283, -119.1113739, 119.1113739
3: -51.9186134, 109.6377792, -51.9186134, 109.6377792, -161.5563812, 161.5563965
4: -26.7571697, 97.5490417, -26.7571697, 97.5490417, -124.3062134, 124.3062134

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5552520
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5552520
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -118.9942169, 249.8049927, -114.1926956, 246.5340118, -365.5281982, 363.9976807
1: -45.3099556, 95.0670929, -44.4512787, 93.1841278, -138.4940796, 139.5183716
2: -21.4827385, 100.0748596, -20.6168480, 98.9276886, -120.4104309, 120.6916885
3: -53.0449715, 111.8905563, -51.7891426, 109.1798248, -162.2247925, 163.6797028
4: -27.3555832, 99.4996338, -26.3381519, 98.1951828, -125.5507660, 125.8377838

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5147223, upper bound: 320.4382549
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5147223, upper bound: 320.4911539
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -118.6119080, 249.1707916, -119.3446579, 256.1723328, -374.7842407, 368.5154419
1: -45.1916962, 94.8024597, -46.2153854, 97.0410690, -142.2327423, 141.0178528
2: -21.4211235, 99.8140030, -21.4967499, 103.0766983, -124.4978180, 121.3107529
3: -52.8986740, 111.5848312, -53.8292046, 113.6120605, -166.5107422, 165.4140320
4: -27.2774563, 99.2399597, -27.3979034, 102.3401566, -129.6176147, 126.6378555

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4654940, upper bound: 320.3158687
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5435815, upper bound: 320.5186600
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -112.7593689, 237.3021851, -137.4283752, 279.6803589, -392.4397278, 374.7305603
1: -43.0751953, 90.4483643, -50.7211494, 106.4939728, -149.5691681, 141.1695099
2: -20.3883018, 95.0837097, -24.5352650, 111.7896576, -132.1779633, 119.6189651
3: -50.4758568, 106.3556976, -59.7634773, 125.2079468, -175.6837921, 166.1191711
4: -26.0264091, 94.5176010, -30.9812927, 111.3961029, -137.4225006, 125.4988937

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5409405, upper bound: 320.5478957
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5409405, upper bound: 320.5478957
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -116.2315903, 244.8721466, -119.8622437, 255.2969666, -371.5285034, 364.7343445
1: -44.3751526, 93.1720428, -46.0409851, 96.4373856, -140.8125305, 139.2130280
2: -20.9795532, 98.1318283, -21.5998268, 102.2101974, -123.1897354, 119.7316513
3: -51.9186134, 109.6377792, -53.7174530, 113.2744598, -165.1930695, 163.3552094
4: -26.7571697, 97.5490417, -27.4959431, 101.5739822, -128.3311462, 125.0449829

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5478503, upper bound: 320.5542314
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5478503, upper bound: 320.5543632
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -139.4374542, 283.2442322, -130.6248322, 261.6815491, -401.1189880, 413.8690796
1: -51.3943214, 107.9217072, -48.0972710, 101.4752731, -152.8695831, 156.0189667
2: -24.9015865, 113.2208786, -23.4330807, 105.5074158, -130.4089966, 136.6539459
3: -60.6053391, 126.9218979, -57.0695915, 119.3523636, -179.9576721, 183.9914856
4: -31.4384499, 112.8451767, -29.7481728, 105.0695648, -136.5080109, 142.5933380

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5372468, upper bound: 320.5346549
time: 0.57 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -139.4374542, 283.2442322, -129.5886536, 265.9031067, -405.3405762, 412.8328857
1: -51.3943214, 107.9217072, -48.3899651, 101.6564026, -153.0507202, 156.3116760
2: -24.9015865, 113.2208786, -23.2732468, 106.3850327, -131.2866211, 136.4941254
3: -60.6053391, 126.9218979, -56.9119797, 119.9124374, -180.5177612, 183.8338776
4: -31.4384499, 112.8451767, -29.5743103, 105.9737320, -137.4121704, 142.4194946

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5372468, upper bound: 320.5351814
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -123.1938248, 260.6712341, -130.6248322, 261.6815491, -384.8753662, 391.2960815
1: -47.0105515, 98.5678024, -48.0972710, 101.4752731, -148.4858093, 146.6650696
2: -22.1759129, 104.2996216, -23.4330807, 105.5074158, -127.6833267, 127.7327042
3: -54.9006386, 115.8525772, -57.0695915, 119.3523636, -174.2529602, 172.9221497
4: -28.1930389, 103.7236404, -29.7481728, 105.0695648, -133.2626038, 133.4717865

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5505039, upper bound: 320.5407818
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5560336, upper bound: 320.5489500
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -123.1938248, 260.6712341, -129.5886536, 265.9031067, -389.0969238, 390.2598877
1: -47.0105515, 98.5678024, -48.3899651, 101.6564026, -148.6669464, 146.9577637
2: -22.1759129, 104.2996216, -23.2732468, 106.3850327, -128.5609436, 127.5728683
3: -54.9006386, 115.8525772, -56.9119797, 119.9124374, -174.8130341, 172.7645416
4: -28.1930389, 103.7236404, -29.5743103, 105.9737320, -134.1667786, 133.2979279

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5505039, upper bound: 320.5482537
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5560336, upper bound: 320.5502448
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -139.4374542, 283.2442322, -139.4374542, 283.2442322, -422.6817017, 422.6817017
1: -51.3943214, 107.9217072, -51.3943214, 107.9217072, -159.3160095, 159.3160248
2: -24.9015865, 113.2208786, -24.9015865, 113.2208786, -138.1224670, 138.1224670
3: -60.6053391, 126.9218979, -60.6053391, 126.9218979, -187.5272369, 187.5272369
4: -31.4384499, 112.8451767, -31.4384499, 112.8451767, -144.2836304, 144.2836304

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5274636, upper bound: 320.5310973
time: 0.60 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5270779, upper bound: 320.5270779
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -139.4374542, 283.2442322, -123.1938248, 260.6712341, -400.1087036, 406.4380493
1: -51.3943214, 107.9217072, -47.0105515, 98.5678024, -149.9621277, 154.9322510
2: -24.9015865, 113.2208786, -22.1759129, 104.2996216, -129.2012024, 135.3967743
3: -60.6053391, 126.9218979, -54.9006386, 115.8525772, -176.4578705, 181.8225403
4: -31.4384499, 112.8451767, -28.1930389, 103.7236404, -135.1620941, 141.0382080

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5274636, upper bound: 320.5310973
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5270779, upper bound: 320.5275875
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -123.1938248, 260.6712341, -139.4374542, 283.2442322, -406.4380493, 400.1087036
1: -47.0105515, 98.5678024, -51.3943214, 107.9217072, -154.9322510, 149.9621277
2: -22.1759129, 104.2996216, -24.9015865, 113.2208786, -135.3967743, 129.2012024
3: -54.9006386, 115.8525772, -60.6053391, 126.9218979, -181.8225403, 176.4578705
4: -28.1930389, 103.7236404, -31.4384499, 112.8451767, -141.0382080, 135.1620941

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5405412, upper bound: 320.5390786
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5410340, upper bound: 320.5461914
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -123.1938248, 260.6712341, -123.1938248, 260.6712341, -383.8650513, 383.8650513
1: -47.0105515, 98.5678024, -47.0105515, 98.5678024, -145.5783234, 145.5783386
2: -22.1759129, 104.2996216, -22.1759129, 104.2996216, -126.4755325, 126.4755325
3: -54.9006386, 115.8525772, -54.9006386, 115.8525772, -170.7531433, 170.7531433
4: -28.1930389, 103.7236404, -28.1930389, 103.7236404, -131.9166718, 131.9166718

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5405412, upper bound: 320.5480084
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5410340, upper bound: 320.5495074
time: 0.65 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.56 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5312137, upper bound: 320.5415570
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5301496, upper bound: 320.5301496
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5312137, upper bound: 320.5438376
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5301496, upper bound: 320.5324303
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5529707
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5529707
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5552520
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5552520
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5147223, upper bound: 320.4382549
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5147223, upper bound: 320.4911539
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.4654940, upper bound: 320.3158687
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5435815, upper bound: 320.5186600
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5409405, upper bound: 320.5478957
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5409405, upper bound: 320.5478957
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5478503, upper bound: 320.5542314
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5478503, upper bound: 320.5543632
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5372468, upper bound: 320.5346549
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5372468, upper bound: 320.5351814
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5505039, upper bound: 320.5407818
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5560336, upper bound: 320.5489500
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5505039, upper bound: 320.5482537
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5560336, upper bound: 320.5502448
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5274636, upper bound: 320.5310973
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5270779, upper bound: 320.5270779
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5274636, upper bound: 320.5310973
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5270779, upper bound: 320.5275875
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5405412, upper bound: 320.5390786
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5410340, upper bound: 320.5461914
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5405412, upper bound: 320.5480084
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -320.5410340, upper bound: 320.5495074

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -115.8823242, 237.8707428, -115.9723358, 243.0606537, -358.9429626, 353.8430786
1: -43.6670837, 91.5956421, -44.1644287, 92.5521393, -136.2192230, 135.7600708
2: -20.9878941, 95.7029953, -20.9728165, 97.3367844, -118.3246765, 116.6758118
3: -51.6053810, 107.4467545, -51.8098679, 108.9799957, -160.5853729, 159.2566223
4: -26.6503849, 95.2635040, -26.7261410, 96.7873306, -123.4377136, 121.9896469

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5301496, upper bound: 320.5301496
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5301496, upper bound: 320.5301496
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -117.7745667, 247.6864777, -119.1956711, 250.1245117, -367.8990784, 366.8821411
1: -44.9041557, 94.1966019, -45.3708763, 95.1940155, -140.0981750, 139.5674744
2: -21.2690449, 99.2145691, -21.5172729, 100.2001801, -121.4692230, 120.7318192
3: -52.5429115, 110.8549347, -53.1180153, 112.0457077, -164.5886230, 163.9729462
4: -27.0793648, 98.6387329, -27.3980484, 99.6261292, -126.7054901, 126.0367737

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5301496, upper bound: 320.5301496
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5301496, upper bound: 320.5301496
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -115.8823242, 237.8707428, -112.7593689, 237.3021851, -353.1845093, 350.6301270
1: -43.6670837, 91.5956421, -43.0751953, 90.4483643, -134.1154480, 134.6708374
2: -20.9878941, 95.7029953, -20.3883018, 95.0837097, -116.0715942, 116.0912933
3: -51.6053810, 107.4467545, -50.4758568, 106.3556976, -157.9610748, 157.9226074
4: -26.6503849, 95.2635040, -26.0264091, 94.5176010, -121.1679764, 121.2899094

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5529707, upper bound: 320.5324303
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5529707, upper bound: 320.5324303
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -117.7745667, 247.6864777, -116.2315903, 244.8721466, -362.6466980, 363.9179993
1: -44.9041557, 94.1966019, -44.3751526, 93.1720428, -138.0761871, 138.5717163
2: -21.2690449, 99.2145691, -20.9795532, 98.1318283, -119.4008713, 120.1941071
3: -52.5429115, 110.8549347, -51.9186134, 109.6377792, -162.1806488, 162.7735443
4: -27.0793648, 98.6387329, -26.7571697, 97.5490417, -124.6284027, 125.3959045

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5529707, upper bound: 320.5324303
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5529707, upper bound: 320.5324303
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -119.8255692, 242.8959045, -115.9723358, 243.0606537, -362.8862305, 358.8682251
1: -44.5374222, 93.9045486, -44.1644287, 92.5521393, -137.0895691, 138.0689697
2: -21.4947472, 97.9566879, -20.9728165, 97.3367844, -118.8315277, 118.9295044
3: -52.5750732, 110.2276993, -51.8098679, 108.9799957, -161.5550537, 162.0375366
4: -27.2980270, 97.3647766, -26.7261410, 96.7873306, -124.0853577, 124.0909195

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5529707
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5529707
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -114.9577484, 242.6368256, -119.1956711, 250.1245117, -365.0822449, 361.8324890
1: -43.9434662, 92.2270279, -45.3708763, 95.1940155, -139.1374817, 137.5979004
2: -20.7524948, 97.2254944, -21.5172729, 100.2001801, -120.9526749, 118.7427597
3: -51.3890228, 108.5482407, -53.1180153, 112.0457077, -163.4347229, 161.6662598
4: -26.4664993, 96.6427689, -27.3980484, 99.6261292, -126.0926208, 124.0408096

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5529707
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5529707
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -119.8255692, 242.8959045, -112.7593689, 237.3021851, -357.1277466, 355.6552734
1: -44.5374222, 93.9045486, -43.0751953, 90.4483643, -134.9857788, 136.9797363
2: -21.4947472, 97.9566879, -20.3883018, 95.0837097, -116.5784454, 118.3449860
3: -52.5750732, 110.2276993, -50.4758568, 106.3556976, -158.9307556, 160.7035522
4: -27.2980270, 97.3647766, -26.0264091, 94.5176010, -121.8156281, 123.3911819

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5552520, upper bound: 320.5552520
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5552520, upper bound: 320.5552520
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -114.9577484, 242.6368256, -116.2315903, 244.8721466, -359.8298340, 358.8683777
1: -43.9434662, 92.2270279, -44.3751526, 93.1720428, -137.1154785, 136.6021576
2: -20.7524948, 97.2254944, -20.9795532, 98.1318283, -118.8843155, 118.2050323
3: -51.3890228, 108.5482407, -51.9186134, 109.6377792, -161.0267639, 160.4668579
4: -26.4664993, 96.6427689, -26.7571697, 97.5490417, -124.0155334, 123.3999329

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5552520, upper bound: 320.5552520
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5552520, upper bound: 320.5552520
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -113.1041260, 240.4769592, -114.1926956, 246.5340118, -359.6380005, 354.6696472
1: -43.5323563, 91.3532257, -44.4512787, 93.1841278, -136.7164917, 135.8045044
2: -20.4742203, 96.4012146, -20.6168480, 98.9276886, -119.4019089, 117.0180511
3: -50.9165306, 107.3582611, -51.7891426, 109.1798248, -160.0963593, 159.1473999
4: -26.1219254, 95.7894974, -26.3381519, 98.1951828, -124.3171082, 122.1276474

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4896933, upper bound: 320.3936544
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5133823, upper bound: 320.4317233
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -119.4280853, 252.8264465, -114.1926956, 246.5340118, -365.9620361, 367.0191345
1: -45.7701302, 96.3067551, -44.4512787, 93.1841278, -138.9542389, 140.7580261
2: -21.5505981, 101.7396011, -20.6168480, 98.9276886, -120.4782867, 122.3564453
3: -53.5379562, 112.9681549, -51.7891426, 109.1798248, -162.7177734, 164.7572784
4: -27.4519711, 101.1291199, -26.3381519, 98.1951828, -125.6471558, 127.4672699

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4896933, upper bound: 320.4368039
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5133823, upper bound: 320.4885550
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -112.0594635, 238.9245148, -117.4498215, 253.1808014, -365.2402649, 356.3743286
1: -43.2244949, 90.6673355, -45.6648979, 95.8856583, -139.1101532, 136.3322296
2: -20.3166142, 95.7305374, -21.1701450, 101.9264679, -122.2430725, 116.9006653
3: -50.5500183, 106.6070633, -53.1673355, 112.2124939, -162.7624969, 159.7743988
4: -25.9290142, 95.1490021, -27.0090675, 101.1777573, -127.1067657, 122.1580658

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4508737, upper bound: 320.2437079
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4625156, upper bound: 320.2883431
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -114.7465134, 244.7402954, -118.1314240, 254.0989838, -368.8454895, 362.8717041
1: -44.2687378, 92.9273987, -45.8287125, 96.2128372, -140.4815674, 138.7561035
2: -20.7197952, 98.3301239, -21.2852840, 102.2361679, -122.9559631, 119.6154099
3: -51.6531792, 109.0710754, -53.3598518, 112.6312103, -164.2843933, 162.4309235
4: -26.4316235, 97.6080399, -27.1395035, 101.4851685, -127.9167938, 124.7475433

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4976784, upper bound: 320.4587939
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5431201, upper bound: 320.5172845
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -119.8255692, 242.8959045, -137.4283752, 279.6803589, -399.5059204, 380.3242798
1: -44.5374222, 93.9045486, -50.7211494, 106.4939728, -151.0314026, 144.6257019
2: -21.4947472, 97.9566879, -24.5352650, 111.7896576, -133.2844086, 122.4919510
3: -52.5750732, 110.2276993, -59.7634773, 125.2079468, -177.7830048, 169.9911652
4: -27.2980270, 97.3647766, -30.9812927, 111.3961029, -138.6941071, 128.3460693

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5339612, upper bound: 320.5367362
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5364041
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -114.9577484, 242.6368256, -137.4283752, 279.6803589, -394.6380615, 380.0651855
1: -43.9434662, 92.2270279, -50.7211494, 106.4939728, -150.4374237, 142.9481812
2: -20.7524948, 97.2254944, -24.5352650, 111.7896576, -132.5421448, 121.7607574
3: -51.3890228, 108.5482407, -59.7634773, 125.2079468, -176.5969391, 168.3117218
4: -26.4664993, 96.6427689, -30.9812927, 111.3961029, -137.8625793, 127.6240616

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5339612, upper bound: 320.5367362
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5364041
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -119.8255692, 242.8959045, -119.8622437, 255.2969666, -375.1225281, 362.7581482
1: -44.5374222, 93.9045486, -46.0409851, 96.4373856, -140.9748077, 139.9455261
2: -21.4947472, 97.9566879, -21.5998268, 102.2101974, -123.7049408, 119.5565186
3: -52.5750732, 110.2276993, -53.7174530, 113.2744598, -165.8495331, 163.9451447
4: -27.2980270, 97.3647766, -27.4959431, 101.5739822, -128.8719940, 124.8607178

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5339612, upper bound: 320.5476990
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5473223
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -114.9577484, 242.6368256, -119.8622437, 255.2969666, -370.2546692, 362.4990540
1: -43.9434662, 92.2270279, -46.0409851, 96.4373856, -140.3808594, 138.2680054
2: -20.7524948, 97.2254944, -21.5998268, 102.2101974, -122.9626846, 118.8253174
3: -51.3890228, 108.5482407, -53.7174530, 113.2744598, -164.6634827, 162.2656860
4: -26.4664993, 96.6427689, -27.4959431, 101.5739822, -128.0404816, 124.1387100

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5339612, upper bound: 320.5468998
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5462462
time: 0.63 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -135.1379395, 275.9418945, -129.2275391, 259.2489929, -394.3869324, 405.1693726
1: -50.0127869, 104.9753265, -47.6405296, 100.4873276, -150.5001068, 152.6158447
2: -24.1440735, 110.2966232, -23.1851959, 104.5302887, -128.6743622, 133.4817963
3: -58.8834152, 123.3897781, -56.4995232, 118.1751251, -177.0585175, 179.8892670
4: -30.4988880, 109.8753891, -29.4352798, 104.0827866, -134.5816498, 139.3106232

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
time: 0.55 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -138.5766907, 281.6685486, -129.8627625, 260.3358154, -398.9125061, 411.5313110
1: -51.1334991, 107.4319458, -47.8427315, 100.9267273, -152.0602264, 155.2746582
2: -24.7447186, 112.6393585, -23.2981415, 104.9644928, -129.7091980, 135.9375000
3: -60.2933998, 126.2360535, -56.7533340, 118.7011108, -178.9944458, 182.9893799
4: -31.2588539, 112.2457886, -29.5766621, 104.5210876, -135.7799377, 141.8224487

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -135.1379395, 275.9418945, -128.0686798, 263.3136292, -398.4515686, 404.0105591
1: -50.0127869, 104.9753265, -47.8995590, 100.5913010, -150.6040802, 152.8748627
2: -24.1440735, 110.2966232, -23.0036297, 105.3399048, -129.4839783, 133.3002319
3: -58.8834152, 123.3897781, -56.2961082, 118.6426849, -177.5260773, 179.6858826
4: -30.4988880, 109.8753891, -29.2342262, 104.9140320, -135.4129181, 139.1096039

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -138.5766907, 281.6685486, -128.7128143, 264.4075623, -402.9841919, 410.3813477
1: -51.1334991, 107.4319458, -48.1052170, 101.0371170, -152.1706238, 155.5371704
2: -24.7447186, 112.6393585, -23.1172600, 105.7844696, -130.5291748, 135.7566071
3: -60.2933998, 126.2360535, -56.5541229, 119.1773758, -179.4707489, 182.7901764
4: -31.2588539, 112.2457886, -29.3758087, 105.3664856, -136.6253357, 141.6215820

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
time: 0.59 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -117.2402039, 250.9446716, -130.4577789, 261.4048767, -378.6450806, 381.4024658
1: -45.2560272, 94.7724686, -48.0449066, 101.3633881, -146.6194153, 142.8173828
2: -21.1287155, 100.5499954, -23.4041004, 105.3968964, -126.5256119, 123.9540939
3: -52.7614594, 111.2368088, -57.0047874, 119.2180634, -171.9794769, 168.2415924
4: -26.9325218, 99.8797989, -29.7114906, 104.9582901, -131.8907776, 129.5912933

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4918634, upper bound: 320.5171532
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5503222, upper bound: 320.5407314
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -122.3987503, 260.5566711, -129.6026611, 260.0114441, -382.4101868, 390.1593323
1: -47.0158157, 98.7183685, -47.7873230, 100.8042755, -147.8200989, 146.5056915
2: -22.0046215, 104.7087555, -23.2599964, 104.8413010, -126.8459244, 127.9687500
3: -54.8009491, 115.6787262, -56.6863899, 118.5550232, -173.3559723, 172.3651123
4: -27.9982967, 104.0213928, -29.5326080, 104.4021606, -132.4004517, 133.5540009

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5036810, upper bound: 320.5378778
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5559248, upper bound: 320.5488088
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -117.2402039, 250.9446716, -129.3851776, 265.5740967, -382.8143005, 380.3298340
1: -45.2560272, 94.7724686, -48.3277092, 101.5226135, -146.7786407, 143.1001740
2: -21.1287155, 100.5499954, -23.2379131, 106.2548599, -127.3835678, 123.7879105
3: -52.7614594, 111.2368088, -56.8340721, 119.7523193, -172.5137634, 168.0708771
4: -26.9325218, 99.8797989, -29.5298080, 105.8423233, -132.7748260, 129.4096069

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5396788, upper bound: 320.5341496
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5337115, upper bound: 320.5465342
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5337115, upper bound: 320.5482537
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -122.3987503, 260.5566711, -128.9998627, 264.9169617, -387.3157043, 389.5565186
1: -47.0158157, 98.7183685, -48.2051430, 101.2493744, -148.2651825, 146.9235077
2: -22.0046215, 104.7087555, -23.1743889, 105.9819031, -127.9865265, 127.8831482
3: -54.8009491, 115.6787262, -56.6823120, 119.4335098, -174.2344666, 172.3610382
4: -27.9982967, 104.0213928, -29.4496689, 105.5697327, -133.5680237, 133.4710693

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.3807316, upper bound: 320.4754450
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5562446, upper bound: 320.5501857
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -135.1379395, 275.9418945, -138.1452332, 281.0290222, -416.1669617, 414.0870972
1: -50.0127869, 104.9753265, -50.9758263, 107.0242081, -157.0369873, 155.9511566
2: -24.1440735, 110.2966232, -24.6735249, 112.3301926, -136.4742432, 134.9701385
3: -58.8834152, 123.3897781, -60.0842781, 125.8489838, -184.7323761, 183.4740448
4: -30.4988880, 109.8753891, -31.1528091, 111.9434433, -142.4423218, 141.0281677

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5270779, upper bound: 320.5270779
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5270779, upper bound: 320.5270779
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -138.5766907, 281.6685486, -138.4962921, 281.5915833, -420.1682739, 420.1647949
1: -51.1334991, 107.4319458, -51.0901680, 107.2672882, -158.4007568, 158.5221100
2: -24.7447186, 112.6393585, -24.7344170, 112.5648041, -137.3094940, 137.3737793
3: -60.2933998, 126.2360535, -60.2226906, 126.1350708, -186.4284363, 186.4587402
4: -31.2588539, 112.2457886, -31.2302494, 112.1731796, -143.4320374, 143.4760132

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5270779, upper bound: 320.5270779
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5270779, upper bound: 320.5270779
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -135.1379395, 275.9418945, -121.6515503, 258.0718079, -393.2097473, 397.5934448
1: -50.0127869, 104.9753265, -46.5324631, 97.5232468, -147.5360413, 151.5077820
2: -24.1440735, 110.2966232, -21.9047222, 103.2699585, -127.4140320, 132.2013397
3: -58.8834152, 123.3897781, -54.3184586, 114.5988770, -173.4822845, 177.7082214
4: -30.4988880, 109.8753891, -27.8606281, 102.6678314, -133.1667175, 137.7360229

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5353235, upper bound: 320.5275872
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5353235, upper bound: 320.5275875
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -138.5766907, 281.6685486, -122.3578415, 259.2653198, -397.8420105, 404.0263977
1: -51.1334991, 107.4319458, -46.7542648, 98.0030212, -149.1365204, 154.1862030
2: -24.7447186, 112.6393585, -22.0298195, 103.7440872, -128.4888000, 134.6691742
3: -60.2933998, 126.2360535, -54.5869026, 115.1756821, -175.4690704, 180.8229523
4: -31.2588539, 112.2457886, -28.0138359, 103.1544495, -134.4132996, 140.2596130

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5353235, upper bound: 320.5275872
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5353235, upper bound: 320.5275875
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -117.2402039, 250.9446716, -139.2988739, 283.0073547, -400.2475586, 390.2435303
1: -45.2560272, 94.7724686, -51.3496132, 107.8282166, -153.0842285, 146.1220856
2: -21.1287155, 100.5499954, -24.8767433, 113.1273499, -134.2560425, 125.4267426
3: -52.7614594, 111.2368088, -60.5493431, 126.8090668, -179.5705261, 171.7861481
4: -26.9325218, 99.8797989, -31.4075012, 112.7503357, -139.6828308, 131.2872925

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5346580, upper bound: 320.5124686
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5405246, upper bound: 320.5387963
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -122.3987503, 260.5566711, -138.1850128, 281.1718140, -403.5704956, 398.7416687
1: -47.0158157, 98.7183685, -51.0098000, 107.1046829, -154.1204987, 149.7281647
2: -22.0046215, 104.7087555, -24.6831875, 112.4030914, -134.4077148, 129.3919373
3: -54.8009491, 115.6787262, -60.1113052, 125.9346695, -180.7356110, 175.7900391
4: -27.9982967, 104.0213928, -31.1662464, 112.0081253, -140.0064240, 135.1876221

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5370187, upper bound: 320.5332231
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5351102, upper bound: 320.5358220
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5274999, upper bound: 320.5352600
time: 0.59 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -117.2402039, 250.9446716, -122.9978867, 260.3507690, -377.5909729, 373.9425659
1: -45.2560272, 94.7724686, -46.9526253, 98.4421692, -143.6981964, 141.7250977
2: -21.1287155, 100.5499954, -22.1415081, 104.1756668, -125.3043823, 122.6915054
3: -52.7614594, 111.2368088, -54.8302040, 115.7000580, -168.4615173, 166.0670166
4: -26.9325218, 99.8797989, -28.1515617, 103.5963440, -130.5288696, 128.0313568

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5476851, upper bound: 320.5476851
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5476851, upper bound: 320.5480084
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -122.3987503, 260.5566711, -122.5756531, 259.5918579, -381.9906006, 383.1323242
1: -47.0158157, 98.7183685, -46.8131485, 98.1411209, -145.1569366, 145.5315247
2: -22.0046215, 104.7087555, -22.0661049, 103.8712997, -125.8759232, 126.7748566
3: -54.8009491, 115.6787262, -54.6541519, 115.3444824, -170.1454315, 170.3328857
4: -27.9982967, 104.0213928, -28.0584469, 103.2856674, -131.2839661, 132.0798340

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5476851, upper bound: 320.5477749
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5476851, upper bound: 320.5495074
time: 0.65 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.21 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5301496, upper bound: 320.5301496
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5301496, upper bound: 320.5301496
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5301496, upper bound: 320.5301496
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5301496, upper bound: 320.5301496
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5529707, upper bound: 320.5324303
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5529707, upper bound: 320.5324303
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5529707, upper bound: 320.5324303
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5529707, upper bound: 320.5324303
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5529707
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5529707
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5529707
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5324303, upper bound: 320.5529707
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5552520, upper bound: 320.5552520
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5552520, upper bound: 320.5552520
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5552520, upper bound: 320.5552520
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5552520, upper bound: 320.5552520
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.4896933, upper bound: 320.3936544
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5133823, upper bound: 320.4317233
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.4896933, upper bound: 320.4368039
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5133823, upper bound: 320.4885550
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.4508737, upper bound: 320.2437079
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.4625156, upper bound: 320.2883431
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.4976784, upper bound: 320.4587939
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5431201, upper bound: 320.5172845
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5339612, upper bound: 320.5367362
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5364041
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5339612, upper bound: 320.5367362
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5364041
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5339612, upper bound: 320.5476990
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5473223
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5339612, upper bound: 320.5468998
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5462462
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5369837, upper bound: 320.5279517
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.4918634, upper bound: 320.5171532
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5503222, upper bound: 320.5407314
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5036810, upper bound: 320.5378778
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5559248, upper bound: 320.5488088
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5337115, upper bound: 320.5465342
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5337115, upper bound: 320.5482537
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.3807316, upper bound: 320.4754450
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5562446, upper bound: 320.5501857
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5270779, upper bound: 320.5270779
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5270779, upper bound: 320.5270779
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5270779, upper bound: 320.5270779
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5270779, upper bound: 320.5270779
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5353235, upper bound: 320.5275872
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5353235, upper bound: 320.5275875
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5353235, upper bound: 320.5275872
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5353235, upper bound: 320.5275875
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5351102, upper bound: 320.5358220
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5274999, upper bound: 320.5352600
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5476851, upper bound: 320.5476851
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5476851, upper bound: 320.5480084
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5476851, upper bound: 320.5477749
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.21
Output dim: 0, lower bound: -320.5476851, upper bound: 320.5495074

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -115.8823242, 237.8707428, -115.8823242, 237.8707428, -353.7530518, 353.7530518
1: -43.6670837, 91.5956421, -43.6670837, 91.5956421, -135.2627258, 135.2627258
2: -20.9878941, 95.7029953, -20.9878941, 95.7029953, -116.6908875, 116.6908875
3: -51.6053810, 107.4467545, -51.6053810, 107.4467545, -159.0521393, 159.0521393
4: -26.6503849, 95.2635040, -26.6503849, 95.2635040, -121.9138794, 121.9138794

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4802439, upper bound: 320.3822822
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5311747, upper bound: 320.5414608
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -115.8823242, 237.8707428, -117.7745667, 247.6864777, -363.5687866, 355.6453247
1: -43.6670837, 91.5956421, -44.9041557, 94.1966019, -137.8636780, 136.4998016
2: -20.9878941, 95.7029953, -21.2690449, 99.2145691, -120.2024536, 116.9720383
3: -51.6053810, 107.4467545, -52.5429115, 110.8549347, -162.4603119, 159.9896698
4: -26.6503849, 95.2635040, -27.0793648, 98.6387329, -125.2891083, 122.3428650

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4802439, upper bound: 320.3822822
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5311747, upper bound: 320.5414608
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -117.7745667, 247.6864777, -115.8823242, 237.8707428, -355.6452942, 363.5687561
1: -44.9041557, 94.1966019, -43.6670837, 91.5956421, -136.4998016, 137.8636780
2: -21.2690449, 99.2145691, -20.9878941, 95.7029953, -116.9720383, 120.2024536
3: -52.5429115, 110.8549347, -51.6053810, 107.4467545, -159.9896698, 162.4603119
4: -27.0793648, 98.6387329, -26.6503849, 95.2635040, -122.3428650, 125.2891083

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5137860, upper bound: 320.4248802
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5281139, upper bound: 320.5281139
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -117.7745667, 247.6864777, -117.7745667, 247.6864777, -365.4609985, 365.4609985
1: -44.9041557, 94.1966019, -44.9041557, 94.1966019, -139.1007385, 139.1007233
2: -21.2690449, 99.2145691, -21.2690449, 99.2145691, -120.4836121, 120.4836121
3: -52.5429115, 110.8549347, -52.5429115, 110.8549347, -163.3978424, 163.3978424
4: -27.0793648, 98.6387329, -27.0793648, 98.6387329, -125.7180939, 125.7180939

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5137860, upper bound: 320.4350714
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5281139, upper bound: 320.5281139
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -115.8823242, 237.8707428, -119.8255692, 242.8959045, -358.7782288, 357.6963196
1: -43.6670837, 91.5956421, -44.5374222, 93.9045486, -137.5716248, 136.1330566
2: -20.9878941, 95.7029953, -21.4947472, 97.9566879, -118.9445801, 117.1977386
3: -51.6053810, 107.4467545, -52.5750732, 110.2276993, -161.8330688, 160.0218201
4: -26.6503849, 95.2635040, -27.2980270, 97.3647766, -124.0151520, 122.5615234

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4754906, upper bound: 320.3796193
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5539956, upper bound: 320.5437954
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -115.8823242, 237.8707428, -114.9577484, 242.6368256, -358.5191650, 352.8284607
1: -43.6670837, 91.5956421, -43.9434662, 92.2270279, -135.8941040, 135.5391083
2: -20.9878941, 95.7029953, -20.7524948, 97.2254944, -118.2133865, 116.4554825
3: -51.6053810, 107.4467545, -51.3890228, 108.5482407, -160.1536255, 158.8357849
4: -26.6503849, 95.2635040, -26.4664993, 96.6427689, -123.2931442, 121.7300034

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4754906, upper bound: 320.3802290
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5539956, upper bound: 320.5437954
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -117.7745667, 247.6864777, -119.8255692, 242.8959045, -360.6704712, 367.5120544
1: -44.9041557, 94.1966019, -44.5374222, 93.9045486, -138.8087006, 138.7340240
2: -21.2690449, 99.2145691, -21.4947472, 97.9566879, -119.2257309, 120.7093048
3: -52.5429115, 110.8549347, -52.5750732, 110.2276993, -162.7705994, 163.4300079
4: -27.0793648, 98.6387329, -27.2980270, 97.3647766, -124.4441376, 125.9367599

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5277771, upper bound: 320.4281969
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5528009, upper bound: 320.5304989
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -117.7745667, 247.6864777, -114.9577484, 242.6368256, -360.4113770, 362.6441650
1: -44.9041557, 94.1966019, -43.9434662, 92.2270279, -137.1311646, 138.1400299
2: -21.2690449, 99.2145691, -20.7524948, 97.2254944, -118.4945374, 119.9670563
3: -52.5429115, 110.8549347, -51.3890228, 108.5482407, -161.0911407, 162.2439575
4: -27.0793648, 98.6387329, -26.4664993, 96.6427689, -123.7221298, 125.1052246

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5277771, upper bound: 320.4397062
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5528009, upper bound: 320.5304989
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -119.8255692, 242.8959045, -115.8823242, 237.8707428, -357.6963196, 358.7782288
1: -44.5374222, 93.9045486, -43.6670837, 91.5956421, -136.1330566, 137.5716248
2: -21.4947472, 97.9566879, -20.9878941, 95.7029953, -117.1977386, 118.9445801
3: -52.5750732, 110.2276993, -51.6053810, 107.4467545, -160.0218201, 161.8330688
4: -27.2980270, 97.3647766, -26.6503849, 95.2635040, -122.5615234, 124.0151520

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4789761, upper bound: 320.4895151
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5314368, upper bound: 320.5528438
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -119.8255692, 242.8959045, -117.7745667, 247.6864777, -367.5120544, 360.6704712
1: -44.5374222, 93.9045486, -44.9041557, 94.1966019, -138.7340240, 138.8086853
2: -21.4947472, 97.9566879, -21.2690449, 99.2145691, -120.7093048, 119.2257309
3: -52.5750732, 110.2276993, -52.5429115, 110.8549347, -163.4300079, 162.7705994
4: -27.2980270, 97.3647766, -27.0793648, 98.6387329, -125.9367599, 124.4441376

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4789761, upper bound: 320.4895151
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5314368, upper bound: 320.5528438
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -114.9577484, 242.6368256, -115.8823242, 237.8707428, -352.8284302, 358.5191650
1: -43.9434662, 92.2270279, -43.6670837, 91.5956421, -135.5391083, 135.8941040
2: -20.7524948, 97.2254944, -20.9878941, 95.7029953, -116.4554825, 118.2133865
3: -51.3890228, 108.5482407, -51.6053810, 107.4467545, -158.8357849, 160.1536102
4: -26.4664993, 96.6427689, -26.6503849, 95.2635040, -121.7300034, 123.2931366

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5180663, upper bound: 320.5082802
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5304989, upper bound: 320.5528009
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -114.9577484, 242.6368256, -117.7745667, 247.6864777, -362.6441650, 360.4113770
1: -43.9434662, 92.2270279, -44.9041557, 94.1966019, -138.1400299, 137.1311646
2: -20.7524948, 97.2254944, -21.2690449, 99.2145691, -119.9670563, 118.4945374
3: -51.3890228, 108.5482407, -52.5429115, 110.8549347, -162.2439575, 161.0911407
4: -26.4664993, 96.6427689, -27.0793648, 98.6387329, -125.1052246, 123.7221298

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5180663, upper bound: 320.5189970
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5304989, upper bound: 320.5528009
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -119.8255692, 242.8959045, -119.8255692, 242.8959045, -362.7214661, 362.7214661
1: -44.5374222, 93.9045486, -44.5374222, 93.9045486, -138.4419708, 138.4419708
2: -21.4947472, 97.9566879, -21.4947472, 97.9566879, -119.4514313, 119.4514313
3: -52.5750732, 110.2276993, -52.5750732, 110.2276993, -162.8027496, 162.8027649
4: -27.2980270, 97.3647766, -27.2980270, 97.3647766, -124.6628036, 124.6628036

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5277590, upper bound: 320.5017308
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5551649, upper bound: 320.5551649
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -119.8255692, 242.8959045, -114.9577484, 242.6368256, -362.4624023, 357.8536377
1: -44.5374222, 93.9045486, -43.9434662, 92.2270279, -136.7644501, 137.8479919
2: -21.4947472, 97.9566879, -20.7524948, 97.2254944, -118.7202377, 118.7091827
3: -52.5750732, 110.2276993, -51.3890228, 108.5482407, -161.1233063, 161.6167145
4: -27.2980270, 97.3647766, -26.4664993, 96.6427689, -123.9407959, 123.8312683

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5277590, upper bound: 320.5017308
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5551649, upper bound: 320.5551649
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -114.9577484, 242.6368256, -119.8255692, 242.8959045, -357.8536072, 362.4624023
1: -43.9434662, 92.2270279, -44.5374222, 93.9045486, -137.8479919, 136.7644501
2: -20.7524948, 97.2254944, -21.4947472, 97.9566879, -118.7091827, 118.7202377
3: -51.3890228, 108.5482407, -52.5750732, 110.2276993, -161.6166992, 161.1233063
4: -26.4664993, 96.6427689, -27.2980270, 97.3647766, -123.8312683, 123.9407883

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5486102, upper bound: 320.5226388
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5551864, upper bound: 320.5551864
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -114.9577484, 242.6368256, -114.9577484, 242.6368256, -357.5945435, 357.5945435
1: -43.9434662, 92.2270279, -43.9434662, 92.2270279, -136.1704712, 136.1704712
2: -20.7524948, 97.2254944, -20.7524948, 97.2254944, -117.9779816, 117.9779816
3: -51.3890228, 108.5482407, -51.3890228, 108.5482407, -159.9372559, 159.9372559
4: -26.4664993, 96.6427689, -26.4664993, 96.6427689, -123.1092606, 123.1092606

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5486102, upper bound: 320.5297120
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5551864, upper bound: 320.5551864
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -112.7970123, 239.9774170, -104.5765533, 231.0454102, -343.8423767, 344.5539551
1: -43.4409752, 91.1567841, -41.6935768, 87.3181992, -130.7591705, 132.8503571
2: -20.4212914, 96.2058182, -18.9998589, 93.1040573, -113.5253448, 115.2056732
3: -50.8036842, 107.1155014, -48.3202667, 101.9007568, -152.7043915, 155.4357605
4: -26.0574074, 95.5860748, -24.3455467, 92.0863342, -118.1437378, 119.9316177

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4842116, upper bound: 320.3718825
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4842116, upper bound: 320.3936544
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -112.5645599, 239.5178680, -108.4729080, 237.1657562, -349.7303162, 347.9906921
1: -43.3545151, 90.9672318, -42.7498627, 89.6807785, -133.0352936, 133.7170868
2: -20.3793583, 96.0118408, -19.6022911, 95.4658127, -115.8451691, 115.6141205
3: -50.6969452, 106.9029160, -49.6809883, 104.9270859, -155.6240234, 156.5839081
4: -26.0032921, 95.3961639, -25.1169872, 94.6414108, -120.6446991, 120.5131531

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4859548, upper bound: 320.3724842
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4859548, upper bound: 320.4317233
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -119.1176224, 252.3202362, -104.5765533, 231.0454102, -350.1630249, 356.8967896
1: -45.6778030, 96.1069641, -41.6935768, 87.3181992, -132.9960022, 137.8005371
2: -21.4969692, 101.5422287, -18.9998589, 93.1040573, -114.6010284, 120.5420837
3: -53.4242058, 112.7219162, -48.3202667, 101.9007568, -155.3249359, 161.0421753
4: -27.3865433, 100.9235764, -24.3455467, 92.0863342, -119.4728699, 125.2691193

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4898620, upper bound: 320.4011419
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4898620, upper bound: 320.4368039
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -118.9278564, 251.9360352, -108.4729080, 237.1657562, -356.0935669, 360.4089355
1: -45.6054459, 95.9498138, -42.7498627, 89.6807785, -135.2862091, 138.6996765
2: -21.4625187, 101.3789597, -19.6022911, 95.4658127, -116.9283295, 120.9812393
3: -53.3346100, 112.5469818, -49.6809883, 104.9270859, -158.2616882, 162.2279663
4: -27.3419170, 100.7642365, -25.1169872, 94.6414108, -121.9833221, 125.8812256

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4942855, upper bound: 320.4042686
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4942855, upper bound: 320.4885550
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -111.7634811, 238.4456940, -108.3261032, 238.6048279, -350.3682861, 346.7717896
1: -43.1375275, 90.4790268, -43.0750580, 90.4028168, -133.5403442, 133.5540771
2: -20.2656670, 95.5436554, -19.6635857, 96.4471970, -116.7128525, 115.2072449
3: -50.4426003, 106.3739471, -49.9573288, 105.3449249, -155.7875214, 156.3312378
4: -25.8669243, 94.9543839, -25.1396294, 95.4455872, -121.3125076, 120.0939941

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4508737, upper bound: 320.2437079
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4508737, upper bound: 320.2437079
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -111.5449524, 238.0089111, -111.5207596, 243.4896545, -355.0345459, 349.5296631
1: -43.0552139, 90.2997513, -43.9285469, 92.4145508, -135.4697571, 134.2282867
2: -20.2261314, 95.3592911, -20.1349792, 98.4431458, -118.6692810, 115.4942627
3: -50.3404579, 106.1734161, -51.0264854, 107.8497009, -158.1901550, 157.1999054
4: -25.8158817, 94.7736206, -25.7421360, 97.5922852, -123.4081650, 120.5157547

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4625156, upper bound: 320.2883431
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4625156, upper bound: 320.2883431
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -114.4365463, 244.2365723, -109.0210495, 239.5415497, -353.9780884, 353.2575989
1: -44.1790276, 92.7290802, -43.2448158, 90.7254333, -134.9044647, 135.9738922
2: -20.6661873, 98.1341553, -19.7714195, 96.7615967, -117.4277725, 117.9055634
3: -51.5428810, 108.8255081, -50.1517563, 105.7699051, -157.3127899, 158.9772491
4: -26.3663540, 97.4035950, -25.2674065, 95.7663651, -122.1327209, 122.6709976

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4838675, upper bound: 320.3667075
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.4838675, upper bound: 320.4587939
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -114.2374573, 243.8367767, -112.1291809, 244.2430878, -358.4804993, 355.9659424
1: -44.1030388, 92.5674591, -44.0639648, 92.6757812, -136.7788239, 136.6313934
2: -20.6301060, 97.9668045, -20.2259350, 98.6864243, -119.3165207, 118.1927338
3: -51.4486427, 108.6451263, -51.1829414, 108.1963806, -159.6450043, 159.8280029
4: -26.3200912, 97.2397232, -25.8522892, 97.8327408, -124.1528244, 123.0920105

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5129347, upper bound: 320.4308490
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5129347, upper bound: 320.5172845
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -118.5093994, 240.5993347, -133.1311493, 272.4250183, -390.9344177, 373.7304688
1: -44.1026535, 92.9678726, -49.3429146, 103.5658264, -147.6684723, 142.3107910
2: -21.2591190, 97.0316315, -23.7794189, 108.8867035, -130.1458130, 120.8110352
3: -52.0312653, 109.1091766, -58.0635223, 121.6952286, -173.7265015, 167.1726837
4: -27.0046101, 96.4310837, -30.0517597, 108.4456482, -135.4502411, 126.4828339

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5364041
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5364041
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -119.0751343, 241.5813446, -136.5075531, 278.0353699, -397.1105042, 378.0888977
1: -44.2871895, 93.3671951, -50.4423256, 105.9725800, -150.2597656, 143.8095245
2: -21.3617001, 97.4265060, -24.3674355, 111.1762543, -132.5379181, 121.7939377
3: -52.2656517, 109.5890884, -59.4286118, 124.4861526, -176.7518005, 169.0177002
4: -27.1324654, 96.8301086, -30.7910061, 110.7623672, -137.8948364, 127.6211166

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5364041
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5364041
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -113.4623489, 240.1319122, -133.1311493, 272.4250183, -385.8873596, 373.2630310
1: -43.4644089, 91.2086258, -49.3429146, 103.5658264, -147.0301971, 140.5515442
2: -20.4878235, 96.2202454, -23.7794189, 108.8867035, -129.3745270, 119.9996414
3: -50.8061905, 107.3183212, -58.0635223, 121.6952286, -172.5014191, 165.3818359
4: -26.1401329, 95.6198425, -30.0517597, 108.4456482, -134.5857391, 125.6715927

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5364041
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5364041
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -114.0594864, 241.1428833, -136.5075531, 278.0353699, -392.0948486, 377.6504517
1: -43.6565094, 91.6220932, -50.4423256, 105.9725800, -149.6290894, 142.0644226
2: -20.5944977, 96.6302567, -24.3674355, 111.1762543, -131.7707367, 120.9976883
3: -51.0414581, 107.8151932, -59.4286118, 124.4861526, -175.5276184, 167.2437897
4: -26.2710800, 96.0390015, -30.7910061, 110.7623672, -137.0334473, 126.8300095

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5364041
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5273466, upper bound: 320.5364041
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -118.5093994, 240.5993347, -115.1276398, 247.4283295, -365.9377441, 355.7269592
1: -44.1026535, 92.9678726, -44.5872955, 93.3398972, -137.4425507, 137.5551605
2: -21.2591190, 97.0316315, -20.7688828, 99.1363449, -120.3954620, 117.8005066
3: -52.0312653, 109.1091766, -51.9505844, 109.5226593, -161.5539093, 161.0597229
4: -27.0046101, 96.4310837, -26.4814758, 98.4321823, -125.4367905, 122.9125595

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5392907, upper bound: 320.5473223
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5392907, upper bound: 320.5473223
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -119.0751343, 241.5813446, -119.5809250, 254.5749512, -373.6500854, 361.1622620
1: -44.2871895, 93.3671951, -45.9232216, 96.2296448, -140.5168304, 139.2904053
2: -21.3617001, 97.4265060, -21.5410309, 101.8942337, -123.2559357, 118.9675293
3: -52.2656517, 109.5890884, -53.5871315, 112.9578018, -165.2234497, 163.1762085
4: -27.1324654, 96.8301086, -27.4299412, 101.2660446, -128.3985138, 124.2600327

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5392907, upper bound: 320.5473223
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5392907, upper bound: 320.5473223
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -113.4623489, 240.1319122, -115.1276398, 247.4283295, -360.8906860, 355.2595520
1: -43.4644089, 91.2086258, -44.5872955, 93.3398972, -136.8043060, 135.7959290
2: -20.4878235, 96.2202454, -20.7688828, 99.1363449, -119.6241684, 116.9891129
3: -50.8061905, 107.3183212, -51.9505844, 109.5226593, -160.3288422, 159.2689056
4: -26.1401329, 95.6198425, -26.4814758, 98.4321823, -124.5723114, 122.1013184

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5464746, upper bound: 320.5453076
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5443176, upper bound: 320.5211445
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -114.0594864, 241.1428833, -119.5809250, 254.5749512, -368.6344299, 360.7238159
1: -43.6565094, 91.6220932, -45.9232216, 96.2296448, -139.8861542, 137.5453186
2: -20.5944977, 96.6302567, -21.5410309, 101.8942337, -122.4887314, 118.1712799
3: -51.0414581, 107.8151932, -53.5871315, 112.9578018, -163.9992676, 161.4022980
4: -26.2710800, 96.0390015, -27.4299412, 101.2660446, -127.5371246, 123.4689331

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5416798, upper bound: 320.5290440
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -320.5410694, upper bound: 320.5206069
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -135.1379395, 275.9418945, -126.0223083, 253.6782532, -388.8161926, 401.9642029
1: -50.0127869, 104.9753265, -46.6003151, 98.2465744, -148.2593689, 151.5756378
2: -24.1440735, 110.2966232, -22.6177673, 102.3056564, -126.4497299, 132.9143829
3: -58.8834152, 123.3897781, -55.2028389, 115.4975739, -174.3809662, 178.5926208
4: -30.4988880, 109.8753891, -28.7227879, 101.8269424, -132.3258057, 138.5981445

Time for backsubstitution: 2.32 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.94 + 417.60 = 421.55 seconds
