## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 187.722369459961


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
execution time: IAR + RelationalAnalysis = 0.92 + 1.99 = 2.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -187.7280013, upper bound: 187.7280013

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270916, upper bound: 187.7272787
time: 0.72 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7269828, upper bound: 187.7269828
time: 0.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.55 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 0, lower bound: -187.7270916, upper bound: 187.7272787
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 0, lower bound: -187.7269828, upper bound: 187.7269828

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -45.9586601, 176.7053528, -46.6790504, 179.5234222, -225.4820709, 223.3843994
1: -122.0961609, 406.3917542, -123.9645233, 412.9741516, -535.0701904, 530.3562622
2: -175.9852295, 357.0053101, -178.4170685, 362.9018555, -538.8870850, 535.4223022
3: -104.2864380, 428.8208923, -105.8679199, 435.6445923, -539.9310303, 534.6887817
4: -162.5491486, 309.3682861, -164.8648682, 314.4447937, -476.9939575, 474.2331543

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7269828, upper bound: 187.7269828
time: 0.69 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7269828, upper bound: 187.7269828
time: 0.76 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -60.8616714, 232.6543427, -45.4478607, 174.5857849, -235.4474487, 278.1021729
1: -162.8272858, 533.6405640, -120.6160965, 401.6061707, -564.4334717, 654.2565918
2: -238.5046539, 467.0584412, -173.8259735, 353.1335144, -591.6380005, 640.8843994
3: -139.4940491, 566.2991943, -103.0145264, 423.6710205, -563.1650391, 669.3137207
4: -219.2331085, 404.9768677, -160.5531769, 305.9475708, -525.1805420, 565.5300293

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7269828, upper bound: 187.7269828
time: 0.79 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7269828, upper bound: 187.7269828
time: 0.78 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.93 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 0, lower bound: -187.7269828, upper bound: 187.7269828
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 0, lower bound: -187.7269828, upper bound: 187.7269828
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 0, lower bound: -187.7269828, upper bound: 187.7269828
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.93
Output dim: 0, lower bound: -187.7269828, upper bound: 187.7269828

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -45.9586601, 176.7053528, -45.9586601, 176.7053528, -222.6640167, 222.6640167
1: -122.0961609, 406.3917542, -122.0961609, 406.3917542, -528.4879150, 528.4879150
2: -175.9852295, 357.0053101, -175.9852295, 357.0053101, -532.9905396, 532.9905396
3: -104.2864380, 428.8208923, -104.2864380, 428.8208923, -533.1072998, 533.1072998
4: -162.5491486, 309.3682861, -162.5491486, 309.3682861, -471.9174194, 471.9174194

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265731, upper bound: 187.7266601
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7267921, upper bound: 187.7269549
time: 0.80 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -45.9586601, 176.7053528, -60.8616714, 232.6543427, -278.6129761, 237.5670166
1: -122.0961609, 406.3917542, -162.8272858, 533.6405640, -655.7366333, 569.2190552
2: -175.9852295, 357.0053101, -238.5046539, 467.0584412, -643.0437012, 595.5098267
3: -104.2864380, 428.8208923, -139.4940491, 566.2991943, -670.5856323, 568.3149414
4: -162.5491486, 309.3682861, -219.2331085, 404.9768677, -567.5260010, 528.6013794

Time for backsubstitution: 0.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265731, upper bound: 187.7266601
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265731, upper bound: 187.7269549
time: 0.79 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -60.8616714, 232.6543427, -45.9586601, 176.7053528, -237.5670166, 278.6129761
1: -162.8272858, 533.6405640, -122.0961609, 406.3917542, -569.2190552, 655.7365723
2: -238.5046539, 467.0584412, -175.9852295, 357.0053101, -595.5098267, 643.0437012
3: -139.4940491, 566.2991943, -104.2864380, 428.8208923, -568.3149414, 670.5856323
4: -219.2331085, 404.9768677, -162.5491486, 309.3682861, -528.6013794, 567.5260010

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265409, upper bound: 187.7263304
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265409, upper bound: 187.7266285
time: 0.82 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -60.8616714, 232.6543427, -60.8616714, 232.6543427, -293.5160217, 293.5159912
1: -162.8272858, 533.6405640, -162.8272858, 533.6405640, -696.4678345, 696.4678345
2: -238.5046539, 467.0584412, -238.5046539, 467.0584412, -705.5630493, 705.5631104
3: -139.4940491, 566.2991943, -139.4940491, 566.2991943, -705.7932129, 705.7932129
4: -219.2331085, 404.9768677, -219.2331085, 404.9768677, -624.2098999, 624.2098389

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265409, upper bound: 187.7263304
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265409, upper bound: 187.7266285
time: 0.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.76 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -187.7265731, upper bound: 187.7266601
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -187.7267921, upper bound: 187.7269549
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -187.7265731, upper bound: 187.7266601
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -187.7265731, upper bound: 187.7269549
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -187.7265409, upper bound: 187.7263304
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -187.7265409, upper bound: 187.7266285
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -187.7265409, upper bound: 187.7263304
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.76
Output dim: 0, lower bound: -187.7265409, upper bound: 187.7266285

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -39.3698540, 151.8095856, -42.9685249, 165.6684875, -205.0383453, 194.7781067
1: -104.8574905, 349.4969482, -113.8191299, 382.3953552, -487.2528076, 463.3160706
2: -149.2269440, 306.1129456, -161.3773651, 335.8831482, -485.1100464, 467.4902954
3: -89.3461533, 368.8677979, -97.0202026, 402.6114502, -491.9575500, 465.8880005
4: -138.3652649, 265.2014771, -149.8638306, 290.9386902, -429.3039551, 415.0653076

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272290, upper bound: 187.7272101
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272250, upper bound: 187.7272116
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -45.2635574, 173.9946594, -45.5216522, 175.0015411, -220.2650909, 219.5163116
1: -120.1942139, 400.2077942, -120.9000702, 402.5063782, -522.7005615, 521.1078491
2: -173.2861938, 351.7464294, -174.2823029, 353.7013550, -526.9874878, 526.0286865
3: -102.6529922, 422.0612183, -103.2585220, 424.5690002, -527.2219849, 525.3197632
4: -160.0516968, 304.7864380, -160.9744263, 306.4888611, -466.5405273, 465.7608337

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275209, upper bound: 187.7276361
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7277851, upper bound: 187.7277851
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -39.3698540, 151.8095856, -57.5553436, 220.4294586, -259.7993164, 209.3649292
1: -104.8574905, 349.4969482, -153.6616058, 506.8405151, -611.6979980, 503.1585693
2: -149.2269440, 306.1129456, -222.7055817, 443.7768860, -593.0037842, 528.8184204
3: -89.3461533, 368.8677979, -131.5323792, 536.9766235, -626.3227539, 500.4001770
4: -138.3652649, 265.2014771, -205.3801575, 384.6049194, -522.9702148, 470.5816345

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7264068, upper bound: 187.7266601
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7264068, upper bound: 187.7266601
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -45.2635574, 173.9946594, -60.5279198, 231.3638000, -276.6273499, 234.5225830
1: -120.1942139, 400.2077942, -161.9166565, 530.6524658, -650.8466187, 562.1244507
2: -173.2861938, 351.7464294, -237.2075958, 464.5482178, -637.8344116, 588.9539185
3: -102.6529922, 422.0612183, -138.7224426, 563.1594849, -665.8125000, 560.7836914
4: -160.0516968, 304.7864380, -218.0496521, 402.7992554, -562.8508301, 522.8360596

Time for backsubstitution: 0.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7266106, upper bound: 187.7268961
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7266106, upper bound: 187.7269549
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -54.9916878, 210.5031738, -42.9685249, 165.6684875, -220.6601715, 253.4716949
1: -147.2212067, 484.1242065, -113.8191299, 382.3953552, -529.6165161, 597.9433594
2: -213.4515076, 422.3138428, -161.3773651, 335.8831482, -549.3346558, 583.6912231
3: -126.0150757, 513.5614014, -97.0202026, 402.6114502, -528.6264038, 610.5816040
4: -196.8030090, 366.1412048, -149.8638306, 290.9386902, -487.7416687, 516.0050049

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7269062, upper bound: 187.7265710
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7268406, upper bound: 187.7264600
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -60.3448410, 230.6542206, -45.5216522, 175.0015411, -235.3463745, 276.1758728
1: -161.4164276, 529.0264893, -120.9000702, 402.5063782, -563.9227295, 649.9265747
2: -236.4884186, 463.1644897, -174.2823029, 353.7013550, -590.1896362, 637.4467163
3: -138.2999725, 561.4500732, -103.2585220, 424.5690002, -562.8689575, 664.7086182
4: -217.3957825, 401.6008911, -160.9744263, 306.4888611, -523.8845825, 562.5753174

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7269549, upper bound: 187.7267573
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7269503, upper bound: 187.7267464
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -54.9916878, 210.5031738, -57.5553436, 220.4294586, -275.4211426, 268.0585022
1: -147.2212067, 484.1242065, -153.6616058, 506.8405151, -654.0617065, 637.7857666
2: -213.4515076, 422.3138428, -222.7055817, 443.7768860, -657.2283936, 645.0194092
3: -126.0150757, 513.5614014, -131.5323792, 536.9766235, -662.9916382, 645.0937500
4: -196.8030090, 366.1412048, -205.3801575, 384.6049194, -581.4077759, 571.5213623

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7255569, upper bound: 187.7253369
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265409, upper bound: 187.7263304
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -60.3448410, 230.6542206, -60.5279198, 231.3638000, -291.7086487, 291.1821289
1: -161.4164276, 529.0264893, -161.9166565, 530.6524658, -692.0687256, 690.9431152
2: -236.4884186, 463.1644897, -237.2075958, 464.5482178, -701.0366211, 700.3720703
3: -138.2999725, 561.4500732, -138.7224426, 563.1594849, -701.4594116, 700.1724854
4: -217.3957825, 401.6008911, -218.0496521, 402.7992554, -620.1950073, 619.6505127

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7266176, upper bound: 187.7265631
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7255569, upper bound: 187.7266285
time: 0.99 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.90 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7272290, upper bound: 187.7272101
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7272250, upper bound: 187.7272116
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7275209, upper bound: 187.7276361
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7277851, upper bound: 187.7277851
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7264068, upper bound: 187.7266601
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7264068, upper bound: 187.7266601
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7266106, upper bound: 187.7268961
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7266106, upper bound: 187.7269549
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7269062, upper bound: 187.7265710
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7268406, upper bound: 187.7264600
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7269549, upper bound: 187.7267573
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7269503, upper bound: 187.7267464
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7255569, upper bound: 187.7253369
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7265409, upper bound: 187.7263304
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7266176, upper bound: 187.7265631
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -187.7255569, upper bound: 187.7266285

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -37.8925323, 146.2183685, -37.9531326, 145.9580231, -183.8505554, 184.1715088
1: -100.8032608, 336.9139709, -100.7326279, 336.0573730, -436.8606262, 437.6466064
2: -142.8260956, 295.1993713, -143.6240692, 295.5504761, -438.3765869, 438.8234253
3: -85.8620529, 355.3522949, -85.8950500, 354.8439026, -440.7059631, 441.2473450
4: -132.6264801, 255.7524414, -133.1443176, 256.0574341, -388.6838989, 388.8967590

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7271122, upper bound: 187.7270914
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272250, upper bound: 187.7272101
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272250, upper bound: 187.7272101
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -38.2092209, 147.3968201, -41.0566635, 158.4209442, -196.6301422, 188.4534912
1: -101.7051849, 339.4154968, -108.6228333, 365.8541565, -467.5593262, 448.0382996
2: -144.4423676, 297.4627686, -153.5706940, 321.6786804, -466.1210022, 451.0334473
3: -86.6141129, 357.9539795, -92.5266495, 384.6318359, -471.2459412, 450.4805908
4: -134.0571289, 257.6193542, -142.7677765, 278.5103149, -412.5674438, 400.3871460

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7271126, upper bound: 187.7270921
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7268417, upper bound: 187.7266487
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7268221, upper bound: 187.7266037
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -43.5517769, 167.4212341, -42.5767441, 163.6678314, -207.2196045, 209.9979858
1: -115.6631699, 385.1258240, -113.0958710, 376.4888306, -492.1520081, 498.2216492
2: -167.0227814, 338.3860779, -163.5513153, 330.6293030, -497.6520996, 501.9373779
3: -98.8115005, 406.2731018, -96.6557159, 397.3070068, -496.1184998, 502.9288330
4: -154.1961517, 293.2812195, -150.9296722, 286.6405029, -440.8366699, 444.2108765

Time for backsubstitution: 0.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275613, upper bound: 187.7274598
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275415, upper bound: 187.7274767
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -44.4690132, 170.9076233, -44.3972397, 170.4940948, -214.9630890, 215.3048248
1: -118.0922394, 393.0397339, -117.9777298, 391.8412170, -509.9334717, 511.0174561
2: -170.4133301, 345.4778442, -170.5875549, 344.4650574, -514.8783569, 516.0653687
3: -100.8742981, 414.6022034, -100.7611465, 413.6239014, -514.4981689, 515.3632812
4: -157.3463287, 299.3963013, -157.3398438, 298.4986267, -455.8449707, 456.7360840

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7273181, upper bound: 187.7275295
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275437, upper bound: 187.7275437
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -39.3698540, 151.8095856, -54.9916878, 210.5031738, -249.8730316, 206.8012695
1: -104.8574905, 349.4969482, -147.2212067, 484.1242065, -588.9816895, 496.7181396
2: -149.2269440, 306.1129456, -213.4515076, 422.3138428, -571.5407715, 519.5644531
3: -89.3461533, 368.8677979, -126.0150757, 513.5614014, -602.9075317, 494.8828735
4: -138.3652649, 265.2014771, -196.8030090, 366.1412048, -504.5064697, 462.0044556

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7263280, upper bound: 187.7265785
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7262111, upper bound: 187.7263003
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -39.3698540, 151.8095856, -60.3448410, 230.6542206, -270.0240784, 212.1544189
1: -104.8574905, 349.4969482, -161.4164276, 529.0264893, -633.8839722, 510.9133301
2: -149.2269440, 306.1129456, -236.4884186, 463.1644897, -612.3914185, 542.6013794
3: -89.3461533, 368.8677979, -138.2999725, 561.4500732, -650.7962036, 507.1677856
4: -138.3652649, 265.2014771, -217.3957825, 401.6008911, -539.9661865, 482.5972595

Time for backsubstitution: 0.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7263280, upper bound: 187.7265788
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7262111, upper bound: 187.7264329
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -45.2635574, 173.9946594, -54.9916878, 210.5031738, -255.7667236, 228.9863434
1: -120.1942139, 400.2077942, -147.2212067, 484.1242065, -604.3184204, 547.4290161
2: -173.2861938, 351.7464294, -213.4515076, 422.3138428, -595.6000366, 565.1978760
3: -102.6529922, 422.0612183, -126.0150757, 513.5614014, -616.2144165, 548.0762939
4: -160.0516968, 304.7864380, -196.8030090, 366.1412048, -526.1928711, 501.5894165

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265860, upper bound: 187.7268597
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7264429, upper bound: 187.7267395
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -45.2635574, 173.9946594, -60.3448410, 230.6542206, -275.9177856, 234.3394928
1: -120.1942139, 400.2077942, -161.4164276, 529.0264893, -649.2207031, 561.6241455
2: -173.2861938, 351.7464294, -236.4884186, 463.1644897, -636.4506836, 588.2348022
3: -102.6529922, 422.0612183, -138.2999725, 561.4500732, -664.1030884, 560.3612061
4: -160.0516968, 304.7864380, -217.3957825, 401.6008911, -561.6525879, 522.1821899

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265860, upper bound: 187.7269521
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7264429, upper bound: 187.7268957
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -53.9210968, 206.3434601, -41.6215706, 160.4487305, -214.3698273, 247.9650269
1: -144.4794159, 474.4480591, -110.3713074, 370.4023438, -514.8816528, 584.8193359
2: -209.8341827, 413.5048218, -156.9167786, 324.7754822, -534.6096802, 570.4216309
3: -123.6918335, 503.5250549, -94.1179733, 390.2459106, -513.9377441, 597.6430054
4: -193.3616791, 358.5704956, -145.5649719, 281.3639832, -474.7256470, 504.1354675

Time for backsubstitution: 0.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7258962, upper bound: 187.7250844
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7262910, upper bound: 187.7257164
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -53.5296173, 204.9373779, -47.2125702, 181.5798035, -235.1094208, 252.1499481
1: -143.2701569, 471.5481873, -125.9801102, 418.0973511, -561.3674927, 597.5283203
2: -207.7874908, 411.1059875, -181.7905426, 365.8935852, -573.6810913, 592.8964844
3: -122.6571808, 500.2193604, -107.6275101, 442.1552734, -564.8124390, 607.8468628
4: -191.5736847, 356.4377747, -167.7667389, 317.1970520, -508.7706909, 524.2044678

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7258946, upper bound: 187.7250499
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7258946, upper bound: 187.7255873
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -59.2675896, 226.4915619, -44.2442093, 170.0229645, -229.2905273, 270.7357788
1: -158.6616821, 519.3857422, -117.6230621, 391.0684814, -549.7301636, 637.0087280
2: -232.8707581, 454.2781982, -170.0935211, 343.0776062, -575.9483643, 624.3717041
3: -135.9721222, 551.5391846, -100.4998016, 412.7583313, -548.7304688, 652.0390015
4: -213.9861908, 393.9567566, -156.9270325, 297.3351135, -511.3212891, 550.8837891

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7258962, upper bound: 187.7256510
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7260876, upper bound: 187.7255382
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -58.9135704, 225.2034607, -49.9111557, 191.4809875, -250.3945465, 275.1145935
1: -157.5531464, 516.7318115, -133.4652405, 439.6518860, -597.2049561, 650.1970215
2: -230.9498901, 452.2205505, -195.2554474, 384.5386658, -615.4885254, 647.4759521
3: -135.0153809, 548.4196777, -114.2144241, 465.9919739, -601.0073242, 662.6340942
4: -212.2836151, 392.1152649, -179.4372864, 333.5409241, -545.8245239, 571.5524292

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7260511, upper bound: 187.7257111
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7260588, upper bound: 187.7254718
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -54.3567657, 208.0394745, -58.7871857, 225.1367798, -279.4935303, 266.8266602
1: -145.5242004, 478.3475647, -156.8997803, 517.3185425, -662.8427734, 635.2473145
2: -211.1156006, 417.3426819, -227.1134644, 453.5662231, -664.6818237, 644.4561768
3: -124.5716019, 507.3862915, -134.2757721, 547.8079834, -672.3795776, 641.6619873
4: -194.6246338, 361.8371887, -209.5828400, 392.9296570, -587.5541992, 571.4198608

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -54.8283691, 209.8635712, -57.1482620, 218.8337555, -273.6621094, 267.0118103
1: -146.7741394, 482.6498413, -152.5499878, 503.1546021, -649.9286499, 635.1998291
2: -212.8226776, 421.0595093, -221.1427460, 440.6366577, -653.4593506, 642.2022705
3: -125.6358032, 511.9879150, -130.5902557, 533.0487671, -658.6845093, 642.5781250
4: -196.2210388, 365.0516968, -203.9333649, 381.8770447, -578.0980225, 568.9850464

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251872, upper bound: 187.7248830
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253163, upper bound: 187.7254145
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -59.7004089, 228.1547546, -61.6472206, 235.6565399, -295.3568726, 289.8019714
1: -159.6945496, 523.1503906, -164.8668060, 540.0534058, -699.7479248, 688.0170288
2: -234.1173248, 458.1044006, -241.2376709, 473.4978943, -707.6152344, 699.3420410
3: -136.8405457, 555.2199707, -141.2225342, 572.9431763, -709.7836914, 696.4425049
4: -215.1961365, 397.2178345, -221.8626251, 410.3804932, -625.5766602, 619.0803833

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -60.1853218, 230.0290833, -60.0811768, 229.6096497, -289.7949524, 290.1102600
1: -160.9786682, 527.5870972, -160.6882782, 526.6148071, -687.5935059, 688.2752686
2: -235.8699799, 461.9429626, -235.4755402, 461.1264648, -696.9962158, 697.4184570
3: -137.9283447, 559.9050293, -137.6807098, 558.8276367, -696.7559814, 697.5857544
4: -216.8241272, 400.5382996, -216.4487915, 399.8218079, -616.6458740, 616.9870605

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251872, upper bound: 187.7256364
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253010, upper bound: 187.7253010
time: 0.85 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.40 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7272250, upper bound: 187.7272101
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7272250, upper bound: 187.7272101
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7268417, upper bound: 187.7266487
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7268221, upper bound: 187.7266037
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7275613, upper bound: 187.7274598
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7275415, upper bound: 187.7274767
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7273181, upper bound: 187.7275295
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7275437, upper bound: 187.7275437
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7263280, upper bound: 187.7265785
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7262111, upper bound: 187.7263003
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7263280, upper bound: 187.7265788
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7262111, upper bound: 187.7264329
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7265860, upper bound: 187.7268597
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7264429, upper bound: 187.7267395
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7265860, upper bound: 187.7269521
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7264429, upper bound: 187.7268957
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7258962, upper bound: 187.7250844
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7262910, upper bound: 187.7257164
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7258946, upper bound: 187.7250499
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7258946, upper bound: 187.7255873
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7258962, upper bound: 187.7256510
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7260876, upper bound: 187.7255382
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7260511, upper bound: 187.7257111
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7260588, upper bound: 187.7254718
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7251872, upper bound: 187.7248830
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7253163, upper bound: 187.7254145
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7251872, upper bound: 187.7256364
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.40
Output dim: 0, lower bound: -187.7253010, upper bound: 187.7253010

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -34.7435036, 133.5505219, -37.9531326, 145.9580231, -180.7015228, 171.5036621
1: -92.6845169, 306.9375610, -100.7326279, 336.0573730, -428.7418823, 407.6701965
2: -132.5847321, 268.7735901, -143.6240692, 295.5504761, -428.1351929, 412.3976440
3: -79.0203629, 324.8642883, -85.8950500, 354.8439026, -433.8642578, 410.7593384
4: -122.7182083, 232.9616241, -133.1443176, 256.0574341, -378.7755737, 366.1059265

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270097, upper bound: 187.7269944
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270097, upper bound: 187.7272101
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -37.4714851, 144.5845642, -37.9531326, 145.9580231, -183.4295044, 182.5376892
1: -99.7184753, 332.9206543, -100.7326279, 336.0573730, -435.7758179, 433.6532898
2: -141.5643005, 291.9348145, -143.6240692, 295.5504761, -437.1147461, 435.5588684
3: -84.9015961, 350.9170837, -85.8950500, 354.8439026, -439.7454834, 436.8121338
4: -131.3877411, 252.7731171, -133.1443176, 256.0574341, -387.4451904, 385.9173584

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270097, upper bound: 187.7269944
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270097, upper bound: 187.7269944
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -36.9584999, 142.5276184, -40.2819405, 155.4013519, -192.3598480, 182.8095551
1: -98.5143127, 328.2343445, -106.6562500, 358.8982544, -457.4125366, 434.8905945
2: -140.3854828, 287.0241394, -151.0760345, 315.2300415, -455.6155396, 438.1001587
3: -83.9360809, 346.4519958, -90.8755493, 377.4963989, -461.4324646, 437.3275452
4: -130.1169434, 248.6234131, -140.3358612, 272.9444885, -403.0614319, 388.9592896

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7266516, upper bound: 187.7266106
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7266516, upper bound: 187.7266487
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -42.8813171, 164.9896698, -38.9573364, 150.3647003, -193.2459869, 203.9469757
1: -114.9086685, 379.4252014, -103.1522217, 347.5210876, -462.4297180, 482.5774231
2: -165.7913361, 331.0296326, -146.0331726, 304.9740906, -470.7653503, 477.0628052
3: -98.0806580, 401.9184570, -87.8895645, 365.5392761, -463.6199036, 489.8080139
4: -152.9176025, 287.0397034, -135.6641541, 264.0815735, -416.9991455, 422.7038574

Time for backsubstitution: 0.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265246, upper bound: 187.7265246
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7265246, upper bound: 187.7266037
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -38.9500580, 149.3336029, -41.0656929, 157.9817047, -196.9317627, 190.3992920
1: -103.6115494, 342.6817932, -108.9373703, 363.7684021, -467.3799438, 451.6191711
2: -150.4227295, 301.5168152, -156.8603210, 319.6079712, -470.0307007, 458.3771362
3: -88.5410385, 362.4757385, -93.0695953, 383.6357117, -472.1767578, 455.5453491
4: -138.6083069, 261.3842468, -144.9273224, 277.0763550, -415.6846619, 406.3115234

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274940, upper bound: 187.7274462
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274940, upper bound: 187.7274598
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -41.7025681, 160.3939056, -41.4480667, 159.3933105, -201.0958710, 201.8419800
1: -110.6592026, 368.9955750, -110.0425720, 366.6818848, -477.3410645, 479.0381470
2: -159.7060089, 324.6441956, -159.0589752, 322.2700195, -481.9760132, 483.7031860
3: -94.4929962, 388.7875061, -94.0169373, 386.6944580, -481.1874390, 482.8044434
4: -147.4506226, 281.2516174, -146.7832489, 279.3285217, -426.7791443, 428.0348511

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261819, upper bound: 187.7263544
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7275415, upper bound: 187.7274767
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -39.6556969, 152.0230713, -42.6908836, 164.0990601, -203.7547455, 194.7139130
1: -105.4899063, 348.6950073, -113.1899872, 377.5787048, -483.0686035, 461.8849792
2: -153.1255341, 306.8614807, -162.8097534, 332.1647339, -485.2902832, 469.6711731
3: -90.1348724, 368.8551636, -96.6733551, 398.1062622, -488.2411499, 465.5285034
4: -141.0941467, 265.9927368, -150.4857330, 287.8065796, -428.9006958, 416.4783630

Time for backsubstitution: 0.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7268845, upper bound: 187.7263221
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7262646, upper bound: 187.7261360
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -42.6267738, 163.9095459, -43.3946190, 166.6662903, -209.2930603, 207.3041687
1: -113.1070328, 376.9688721, -115.2744675, 382.9899597, -496.0969849, 492.2433472
2: -163.0957794, 331.7691345, -166.6106110, 337.0396729, -500.1354065, 498.3797607
3: -96.5724945, 397.2020569, -98.4219742, 404.0842896, -500.6567993, 495.6240234
4: -150.6133118, 287.3820801, -153.7063446, 291.9806213, -442.5939331, 441.0883789

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7268853, upper bound: 187.7263198
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261437, upper bound: 187.7261437
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -39.1078987, 150.7899475, -54.8876648, 210.0942535, -249.2021179, 205.6775818
1: -104.1526108, 347.1518250, -146.9509888, 483.1636047, -587.3161011, 494.1027832
2: -148.2697906, 304.0660095, -213.1074066, 421.4789124, -569.7486572, 517.1734009
3: -88.7511673, 366.3943787, -125.7866058, 512.5610962, -601.3122559, 492.1809692
4: -137.4653473, 263.4257202, -196.4738922, 365.4194031, -502.8847656, 459.8995972

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7260912, upper bound: 187.7262585
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7260912, upper bound: 187.7263003
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -39.1598473, 151.1481628, -54.0601654, 206.9711456, -246.1309814, 205.2083282
1: -104.3518448, 348.1903381, -144.6987457, 476.1172485, -580.4689331, 492.8890381
2: -148.2908020, 304.5467834, -209.7518158, 415.2625732, -563.5533447, 514.2985840
3: -88.8953400, 367.7827759, -123.8570023, 504.9985657, -593.8938599, 491.6397400
4: -137.5424500, 263.9661865, -193.4004364, 360.0197144, -497.5621643, 457.3666077

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7260912, upper bound: 187.7262585
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7260912, upper bound: 187.7263003
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -39.1078987, 150.7899475, -60.2355309, 230.2280731, -269.3359375, 211.0254517
1: -104.1526108, 347.1518250, -161.1311340, 528.0384521, -632.1910400, 508.2829590
2: -148.2697906, 304.0660095, -236.1173096, 462.2902222, -610.5599976, 540.1833496
3: -88.7511673, 366.3943787, -138.0582733, 560.4226074, -649.1737061, 504.4525757
4: -137.4653473, 263.4257202, -217.0467987, 400.8444824, -538.3098145, 480.4725342

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259837, upper bound: 187.7261557
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259837, upper bound: 187.7264329
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -39.1598473, 151.1481628, -59.4405060, 227.2168121, -266.3766479, 210.5886688
1: -104.3518448, 348.1903381, -158.9693146, 521.2439575, -625.5958252, 507.1596069
2: -148.2908020, 304.5467834, -232.8878937, 456.3361206, -604.6268311, 537.4346924
3: -88.8953400, 367.7827759, -136.2007446, 553.1198730, -642.0151978, 503.9835205
4: -137.5424500, 263.9661865, -214.0801544, 395.6719360, -533.2143555, 478.0463257

Time for backsubstitution: 0.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259837, upper bound: 187.7261557
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259837, upper bound: 187.7264329
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -44.9962158, 172.9530029, -54.8876648, 210.0942535, -255.0904541, 227.8406372
1: -119.4690475, 397.8204956, -146.9509888, 483.1636047, -602.6325073, 544.7714844
2: -172.2857513, 349.6840820, -213.1074066, 421.4789124, -593.7646484, 562.7915039
3: -102.0392227, 419.5346069, -125.7866058, 512.5610962, -614.6003418, 545.3212280
4: -159.1103210, 302.9974670, -196.4738922, 365.4194031, -524.5296021, 499.4713135

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7262746, upper bound: 187.7266320
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7262746, upper bound: 187.7267395
time: 1.35 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -45.3276634, 174.3791504, -54.0601654, 206.9711456, -252.2987976, 228.4392853
1: -120.3909531, 401.4574585, -144.6987457, 476.1172485, -596.5081787, 546.1560669
2: -173.2749176, 352.4047852, -209.7518158, 415.2625732, -588.5374756, 562.1565552
3: -102.7993240, 423.6196594, -123.8570023, 504.9985657, -607.7979126, 547.4765625
4: -160.1105652, 305.4512634, -193.4004364, 360.0197144, -520.1302490, 498.8516235

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7262746, upper bound: 187.7266320
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7262746, upper bound: 187.7267395
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -44.9962158, 172.9530029, -60.2355309, 230.2280731, -275.2242737, 233.1885223
1: -119.4690475, 397.8204956, -161.1311340, 528.0384521, -647.5075073, 558.9515381
2: -172.2857513, 349.6840820, -236.1173096, 462.2902222, -634.5759888, 585.8013916
3: -102.0392227, 419.5346069, -138.0582733, 560.4226074, -662.4618530, 557.5928955
4: -159.1103210, 302.9974670, -217.0467987, 400.8444824, -559.9548340, 520.0441284

Time for backsubstitution: 0.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7262126, upper bound: 187.7265933
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7262126, upper bound: 187.7268957
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -45.3276634, 174.3791504, -59.4405060, 227.2168121, -272.5444641, 233.8196411
1: -120.3909531, 401.4574585, -158.9693146, 521.2439575, -641.6348877, 560.4266968
2: -173.2749176, 352.4047852, -232.8878937, 456.3361206, -629.6110229, 585.2926025
3: -102.7993240, 423.6196594, -136.2007446, 553.1198730, -655.9191895, 559.8204346
4: -160.1105652, 305.4512634, -214.0801544, 395.6719360, -555.7824097, 519.5314331

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7262126, upper bound: 187.7265933
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7262126, upper bound: 187.7268957
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -52.0224991, 198.8103943, -39.1519890, 150.8533630, -202.8758545, 237.9623718
1: -139.0080414, 456.3614197, -103.7259369, 348.0252380, -487.0332642, 560.0872803
2: -201.8192444, 399.6740723, -147.6744080, 305.7457275, -507.5649719, 547.3485107
3: -119.0109406, 483.8226318, -88.4269943, 366.3183899, -485.3293152, 572.2495728
4: -186.0676880, 346.3368225, -136.9807892, 264.7182922, -450.7859497, 483.3176270

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7258139, upper bound: 187.7250737
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7258139, upper bound: 187.7250844
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -52.8027420, 202.0806122, -41.3027534, 159.2343140, -212.0370483, 243.3833618
1: -141.4395294, 464.6400757, -109.4932251, 367.6607056, -509.1001587, 574.1333008
2: -205.3082581, 405.1242065, -155.5434265, 322.4428406, -527.7510986, 560.6676025
3: -121.0769424, 493.0162964, -93.3587875, 387.2708435, -508.3477783, 586.3750610
4: -189.1998291, 351.2576599, -144.3205872, 279.3189087, -468.5187378, 495.5781860

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7262910, upper bound: 187.7256831
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7254834, upper bound: 187.7249053
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259302, upper bound: 187.7252540
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -51.2129440, 195.7597046, -44.6668129, 171.7068939, -222.9198151, 240.4265137
1: -136.7541504, 449.6931152, -119.1183090, 394.6998596, -531.4539795, 568.8114014
2: -198.4866486, 393.7602844, -172.2701416, 346.3308411, -544.8175049, 566.0303345
3: -117.0897827, 476.6947937, -101.7658615, 417.3779297, -534.4677124, 578.4606323
4: -183.0187836, 341.1972046, -158.9098206, 300.1151733, -483.1339722, 500.1070251

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7256968, upper bound: 187.7249983
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7256968, upper bound: 187.7249983
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -52.3537979, 200.4330750, -46.7758293, 179.9255676, -232.2793579, 247.2089081
1: -140.0747986, 461.2239990, -124.7776260, 414.3747864, -554.4495850, 586.0015259
2: -203.0335999, 402.2585754, -179.9716034, 362.6557312, -565.6892700, 582.2301636
3: -119.9059677, 489.1560364, -106.5927582, 438.1324158, -558.0383301, 595.7487793
4: -187.1989746, 348.7276001, -166.1050262, 314.3786926, -501.5776672, 514.8326416

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7260872, upper bound: 187.7255849
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7256968, upper bound: 187.7255873
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -57.3260422, 218.7723236, -41.8106461, 160.5350037, -217.8610535, 260.5829773
1: -153.0643768, 500.8387146, -111.1057739, 368.8266602, -521.8909912, 611.9443970
2: -224.7106781, 440.1973572, -161.2296906, 324.1296082, -548.8400879, 601.4270630
3: -131.1840057, 531.3409424, -94.9279633, 389.2096558, -520.3936768, 626.2689209
4: -206.5070190, 381.4566650, -148.6014099, 280.7810669, -487.2880859, 530.0581055

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259625, upper bound: 187.7255041
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259625, upper bound: 187.7255222
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -58.1453362, 222.1918030, -43.8985672, 168.7072906, -226.8526001, 266.0903625
1: -155.6022644, 509.4779663, -116.6712723, 388.0997314, -543.7020264, 626.1492310
2: -228.3250580, 445.8673706, -168.6242523, 340.5300293, -568.8551025, 614.4916382
3: -133.3358917, 540.8165283, -99.6786346, 409.5664673, -542.9023438, 640.4951782
4: -209.7920380, 386.6254883, -155.5836945, 295.1115417, -504.9035645, 542.2090454

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259625, upper bound: 187.7255041
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259418, upper bound: 187.7255382
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -56.5777512, 215.9529724, -47.4151230, 181.8184967, -238.3962402, 263.3681030
1: -150.9909363, 494.6424255, -126.8031616, 416.7138672, -567.7048340, 621.4455566
2: -221.6466675, 434.7843933, -186.0067291, 365.2035217, -586.8500366, 620.7911377
3: -129.4141846, 524.7028198, -108.4967346, 441.7203674, -571.1345215, 633.1995239
4: -203.6952209, 376.7477112, -170.8164978, 316.6808777, -520.3760376, 547.5642090

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7258967, upper bound: 187.7256062
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7258967, upper bound: 187.7257111
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -57.7306824, 220.6721039, -49.4315071, 189.6503754, -247.3810577, 270.1035767
1: -154.3366547, 506.2919312, -132.1484222, 435.5017090, -589.8383789, 638.4402466
2: -226.1753998, 443.3265991, -193.2925262, 380.9627991, -607.1381836, 636.6191406
3: -132.2439117, 537.1520996, -113.0779190, 461.5242920, -593.7681274, 650.2299194
4: -207.8765717, 384.3719788, -177.6321869, 330.4326782, -538.3092651, 562.0041504

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259740, upper bound: 187.7254487
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259740, upper bound: 187.7254543
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -52.8892097, 202.1773529, -54.9389725, 210.2270966, -263.1162720, 257.1163025
1: -141.1726074, 464.2333984, -146.5529785, 482.7725525, -623.9451294, 610.7861938
2: -204.5861511, 406.8985596, -212.9181976, 423.4718323, -628.0578613, 619.8167725
3: -120.8497620, 491.9433289, -125.4880905, 511.3842468, -632.2340088, 617.4313965
4: -188.7357483, 352.5541077, -196.2660065, 366.9425049, -555.6782227, 548.8201294

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251816, upper bound: 187.7248755
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251816, upper bound: 187.7248830
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -53.7087860, 205.5821228, -56.6182594, 216.8085175, -270.5173035, 262.2003784
1: -143.7259827, 472.7937317, -151.1054535, 498.4817810, -642.2077637, 623.8991699
2: -208.2811432, 412.6273804, -218.9800262, 436.6610107, -644.9421387, 631.6072998
3: -123.0127182, 501.4205933, -129.3424072, 528.0459595, -651.0586548, 630.7630005
4: -192.0481720, 357.6968994, -201.9429626, 378.4118958, -570.4600220, 559.6398315

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253105, upper bound: 187.7254062
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253105, upper bound: 187.7254145
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -58.1915817, 222.1187592, -57.9264870, 221.2028809, -279.3944702, 280.0452576
1: -155.2480164, 508.5930481, -154.8386841, 506.8444519, -662.0924683, 663.4317017
2: -227.5268250, 447.4207153, -227.4101868, 444.3479919, -671.8748169, 674.8308716
3: -133.0295105, 539.2510376, -132.6923981, 537.8785400, -670.9080811, 671.9432983
4: -209.1822052, 387.6681213, -208.9471283, 385.2294006, -594.4116211, 596.6152344

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253182, upper bound: 187.7255955
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251843, upper bound: 187.7252980
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251843, upper bound: 187.7252981
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -59.0557251, 225.7017212, -59.5182991, 227.4525757, -286.5083008, 285.2199707
1: -157.8997955, 517.5921021, -159.1533356, 521.6341553, -679.5339355, 676.7454224
2: -231.2984924, 453.4600830, -233.1977997, 456.8997498, -688.1982422, 686.6578369
3: -135.2765198, 549.1116333, -136.3597260, 553.4436035, -688.7200928, 685.4713745
4: -212.6048279, 393.1467896, -214.3452606, 396.1393738, -608.7441406, 607.4920044

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252955, upper bound: 187.7252942
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252955, upper bound: 187.7253003
time: 0.88 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.77 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7270097, upper bound: 187.7269944
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7270097, upper bound: 187.7272101
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7270097, upper bound: 187.7269944
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7270097, upper bound: 187.7269944
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7266516, upper bound: 187.7266106
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7266516, upper bound: 187.7266487
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7265246, upper bound: 187.7265246
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7265246, upper bound: 187.7266037
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7274940, upper bound: 187.7274462
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7274940, upper bound: 187.7274598
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7261819, upper bound: 187.7263544
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7275415, upper bound: 187.7274767
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7268845, upper bound: 187.7263221
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7262646, upper bound: 187.7261360
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7268853, upper bound: 187.7263198
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7261437, upper bound: 187.7261437
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7260912, upper bound: 187.7262585
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7260912, upper bound: 187.7263003
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7260912, upper bound: 187.7262585
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7260912, upper bound: 187.7263003
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7259837, upper bound: 187.7261557
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7259837, upper bound: 187.7264329
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7259837, upper bound: 187.7261557
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7259837, upper bound: 187.7264329
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7262746, upper bound: 187.7266320
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7262746, upper bound: 187.7267395
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7262746, upper bound: 187.7266320
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7262746, upper bound: 187.7267395
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7262126, upper bound: 187.7265933
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7262126, upper bound: 187.7268957
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7262126, upper bound: 187.7265933
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7262126, upper bound: 187.7268957
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7258139, upper bound: 187.7250737
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7258139, upper bound: 187.7250844
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7254834, upper bound: 187.7249053
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7259302, upper bound: 187.7252540
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7256968, upper bound: 187.7249983
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7256968, upper bound: 187.7249983
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7260872, upper bound: 187.7255849
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7256968, upper bound: 187.7255873
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7259625, upper bound: 187.7255041
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7259625, upper bound: 187.7255222
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7259625, upper bound: 187.7255041
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7259418, upper bound: 187.7255382
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7258967, upper bound: 187.7256062
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7258967, upper bound: 187.7257111
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7259740, upper bound: 187.7254487
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7259740, upper bound: 187.7254543
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7251816, upper bound: 187.7248755
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7251816, upper bound: 187.7248830
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7253105, upper bound: 187.7254062
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7253105, upper bound: 187.7254145
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7251843, upper bound: 187.7252980
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7251843, upper bound: 187.7252981
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7252955, upper bound: 187.7252942
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.77
Output dim: 0, lower bound: -187.7252955, upper bound: 187.7253003

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -34.7435036, 133.5505219, -34.7435036, 133.5505219, -168.2940216, 168.2940216
1: -92.6845169, 306.9375610, -92.6845169, 306.9375610, -399.6220703, 399.6220703
2: -132.5847321, 268.7735901, -132.5847321, 268.7735901, -401.3583374, 401.3583374
3: -79.0203629, 324.8642883, -79.0203629, 324.8642883, -403.8846436, 403.8846436
4: -122.7182083, 232.9616241, -122.7182083, 232.9616241, -355.6798096, 355.6797791

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270142, upper bound: 187.7270056
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7273414, upper bound: 187.7273414
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -34.7435036, 133.5505219, -40.4066696, 154.8893890, -189.6328888, 173.9571838
1: -92.6845169, 306.9375610, -107.4885941, 355.3005371, -447.9850464, 414.4261475
2: -132.5847321, 268.7735901, -155.9506683, 312.6643066, -445.2490234, 424.7242432
3: -79.0203629, 324.8642883, -91.8419571, 375.8139648, -454.8343201, 416.7062378
4: -122.7182083, 232.9616241, -143.7290649, 270.9963074, -393.7144165, 376.6906738

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270142, upper bound: 187.7270056
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270142, upper bound: 187.7274779
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -37.4714851, 144.5845642, -34.7435036, 133.5505219, -171.0220032, 179.3280640
1: -99.7184753, 332.9206543, -92.6845169, 306.9375610, -406.6560059, 425.6051636
2: -141.5643005, 291.9348145, -132.5847321, 268.7735901, -410.3378906, 424.5195007
3: -84.9015961, 350.9170837, -79.0203629, 324.8642883, -409.7658691, 429.9374390
4: -131.3877411, 252.7731171, -122.7182083, 232.9616241, -364.3493652, 375.4912109

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270037, upper bound: 187.7269866
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270059, upper bound: 187.7269889
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -37.4714851, 144.5845642, -40.4066696, 154.8893890, -192.3608551, 184.9912262
1: -99.7184753, 332.9206543, -107.4885941, 355.3005371, -455.0189514, 440.4092407
2: -141.5643005, 291.9348145, -155.9506683, 312.6643066, -454.2286072, 447.8854370
3: -84.9015961, 350.9170837, -91.8419571, 375.8139648, -460.7155762, 442.7590332
4: -131.3877411, 252.7731171, -143.7290649, 270.9963074, -402.3840332, 396.5021667

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270037, upper bound: 187.7269866
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270059, upper bound: 187.7269889
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -36.9584999, 142.5276184, -36.7249603, 141.6688538, -178.6273346, 179.2525787
1: -98.5143127, 328.2343445, -97.8184967, 326.2103271, -424.7246399, 426.0527954
2: -140.3854828, 287.0241394, -139.1670990, 285.6815186, -426.0670166, 426.1912231
3: -83.9360809, 346.4519958, -83.3096695, 344.0377197, -427.9738159, 429.7615967
4: -130.1169434, 248.6234131, -129.0505219, 247.3886414, -377.5055237, 377.6739502

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261745, upper bound: 187.7263138
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7263128, upper bound: 187.7263813
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -36.9584999, 142.5276184, -42.6744385, 164.0748291, -201.0333252, 185.2020569
1: -98.5143127, 328.2343445, -113.3021698, 377.4269409, -475.9412231, 441.5364990
2: -140.3854828, 287.0241394, -163.5760651, 331.7731628, -472.1586304, 450.6002197
3: -83.9360809, 346.4519958, -96.7497711, 397.7746887, -481.7107544, 443.2017517
4: -130.1169434, 248.6234131, -150.9831696, 287.3827820, -417.4997253, 399.6065674

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261745, upper bound: 187.7264439
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7263128, upper bound: 187.7263813
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -42.8813171, 164.9896698, -35.3544846, 136.4710999, -179.3524017, 200.3441467
1: -114.9086685, 379.4252014, -94.2080841, 314.4536438, -429.3623047, 473.6332397
2: -165.7913361, 331.0296326, -133.9721680, 275.1189575, -440.9102783, 465.0018005
3: -98.0806580, 401.9184570, -80.2256317, 331.6719666, -429.7525330, 482.1441040
4: -152.9176025, 287.0397034, -124.2371216, 238.2488556, -391.1664429, 411.2767334

Time for backsubstitution: 0.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7258972, upper bound: 187.7257444
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7258972, upper bound: 187.7260036
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -42.8813171, 164.9896698, -41.2314987, 158.6071014, -201.4884033, 206.2211609
1: -114.9086685, 379.4252014, -109.5003128, 365.0538330, -479.9624329, 488.9255066
2: -165.7913361, 331.0296326, -158.1243286, 320.6061707, -486.3974609, 489.1539612
3: -98.0806580, 401.9184570, -93.5115662, 384.7761536, -482.8568115, 495.4300232
4: -152.9176025, 287.0397034, -145.9332275, 277.7350159, -430.6526184, 432.9728394

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7258972, upper bound: 187.7258689
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7260036, upper bound: 187.7261290
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -38.9500580, 149.3336029, -38.2038651, 146.4945221, -185.4445801, 187.5374298
1: -103.6115494, 342.6817932, -101.6303711, 336.2579651, -439.8695068, 444.3121033
2: -150.4227295, 301.5168152, -147.6074219, 295.7599182, -446.1826477, 449.1242371
3: -88.5410385, 362.4757385, -86.8700714, 355.7000427, -444.2410889, 449.3457947
4: -138.6083069, 261.3842468, -135.9981079, 256.4407654, -395.0490723, 397.3823547

Time for backsubstitution: 0.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272126, upper bound: 187.7274130
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274130, upper bound: 187.7274471
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -38.9500580, 149.3336029, -40.7154007, 156.5954895, -195.5455475, 190.0489807
1: -103.6115494, 342.6817932, -108.0657349, 360.2446899, -463.8562317, 450.7474976
2: -150.4227295, 301.5168152, -156.2054443, 316.8060913, -467.2288208, 457.7222595
3: -88.5410385, 362.4757385, -92.3120728, 379.7305298, -468.2715759, 454.7878113
4: -138.6083069, 261.3842468, -144.1536102, 274.5439453, -413.1522522, 405.5378418

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274130, upper bound: 187.7274242
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7274129, upper bound: 187.7274598
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -41.0704460, 157.9169922, -42.8632545, 164.9234314, -205.9938660, 200.7802429
1: -109.0113983, 363.0211792, -113.8047333, 378.8442383, -487.8556519, 476.8259277
2: -157.4728394, 319.7153931, -163.9342804, 333.4321899, -490.9050293, 483.6496582
3: -93.0688095, 382.4017944, -97.1969833, 399.3837891, -492.4525757, 479.5987549
4: -145.3551178, 276.9440613, -151.5098114, 288.8169861, -434.1720886, 428.4538574

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261819, upper bound: 187.7263544
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261819, upper bound: 187.7263544
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -41.5595551, 159.8413849, -41.0637817, 157.9071960, -199.4667511, 200.9051514
1: -110.2639771, 367.7517395, -108.9840393, 363.3309021, -473.5948486, 476.7357483
2: -159.1325378, 323.5677795, -157.5265808, 319.3767090, -478.5091553, 481.0943604
3: -94.1566086, 387.4514771, -93.1150665, 383.0951843, -477.2518005, 480.5665283
4: -146.9217224, 280.3191223, -145.3693542, 276.8207703, -423.7424927, 425.6884766

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259058, upper bound: 187.7257098
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259058, upper bound: 187.7274767
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -38.4776573, 147.5130157, -40.8082123, 156.8591614, -195.3368073, 188.3212128
1: -102.3232346, 338.4113159, -108.0886002, 361.0902100, -463.4134216, 446.4999084
2: -148.6835785, 297.8030701, -155.6349030, 317.6912842, -466.3748474, 453.4379883
3: -87.4508972, 357.9538269, -92.3493195, 380.6189270, -468.0697937, 450.3031616
4: -136.9790497, 258.1623230, -143.8379059, 275.2825928, -412.2616272, 402.0002441

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7263462, upper bound: 187.7260203
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7267927, upper bound: 187.7262270
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -38.5951920, 147.9384766, -44.6880608, 171.8832703, -210.4784546, 192.6265411
1: -102.6268234, 339.3081360, -118.3060379, 396.8739319, -499.5007629, 457.6141663
2: -149.1276093, 298.7122192, -169.1756897, 347.9330139, -497.0606079, 467.8878784
3: -87.7034607, 358.9016113, -100.9240112, 419.1083374, -506.8117981, 459.8255920
4: -137.3726196, 258.9380188, -156.4599762, 301.6853638, -439.0579834, 415.3979797

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7262280, upper bound: 187.7259466
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7262427, upper bound: 187.7260482
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -41.5286446, 159.7199097, -41.5706482, 159.6403046, -201.1689453, 201.2905579
1: -110.1400070, 367.4047852, -110.3620911, 366.9570007, -477.0970154, 477.7667847
2: -158.9035187, 323.3880005, -159.7448120, 322.9657898, -481.8692932, 483.1328125
3: -94.0516357, 387.0624695, -94.2594833, 387.0828552, -481.1344299, 481.3219604
4: -146.7330322, 280.1291504, -147.3271942, 279.8028870, -426.5359192, 427.4563599

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7249643, upper bound: 187.7247023
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7268853, upper bound: 187.7263198
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -41.3564415, 158.9084930, -46.0034103, 176.9018250, -218.2582703, 204.9118805
1: -109.7288284, 365.3716431, -121.9962387, 408.1092834, -517.8380737, 487.3678894
2: -158.5905914, 321.6476440, -175.4631958, 357.3945312, -515.9851074, 497.1108398
3: -93.7230301, 385.0737610, -104.0978622, 431.0924988, -524.8155518, 489.1716003
4: -146.3489685, 278.6096497, -161.9647369, 309.9106140, -456.2595520, 440.5744019

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7245619, upper bound: 187.7245473
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261437, upper bound: 187.7261437
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -39.1078987, 150.7899475, -54.7390404, 209.5062256, -248.6141205, 205.5289764
1: -104.1526108, 347.1518250, -146.5640564, 481.7807312, -585.9332886, 493.7158813
2: -148.2697906, 304.0660095, -212.6123810, 420.2774658, -568.5472412, 516.6784058
3: -88.7511673, 366.3943787, -125.4597855, 511.1200562, -599.8712158, 491.8541565
4: -137.4653473, 263.4257202, -196.0008545, 364.3806152, -501.8459473, 459.4265747

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7260942, upper bound: 187.7264565
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259520, upper bound: 187.7262334
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -39.1078987, 150.7899475, -55.0499115, 210.9465637, -250.0544586, 205.8398590
1: -104.1526108, 347.1518250, -147.3358307, 485.7331238, -589.8856812, 494.4875793
2: -148.2697906, 304.0660095, -213.2291412, 423.0662231, -571.3359985, 517.2951660
3: -88.7511673, 366.3943787, -126.0885849, 515.1484985, -603.8995972, 492.4829712
4: -137.4653473, 263.4257202, -196.6598206, 366.8345642, -504.2999268, 460.0855408

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7260942, upper bound: 187.7264570
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259520, upper bound: 187.7263012
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -39.1598473, 151.1481628, -54.7390404, 209.5062256, -248.6660767, 205.8871918
1: -104.3518448, 348.1903381, -146.5640564, 481.7807312, -586.1325073, 494.7543640
2: -148.2908020, 304.5467834, -212.6123810, 420.2774658, -568.5682373, 517.1591797
3: -88.8953400, 367.7827759, -125.4597855, 511.1200562, -600.0153198, 493.2425537
4: -137.5424500, 263.9661865, -196.0008545, 364.3806152, -501.9230652, 459.9670410

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -39.1598473, 151.1481628, -55.0499115, 210.9465637, -250.1064148, 206.1980743
1: -104.3518448, 348.1903381, -147.3358307, 485.7331238, -590.0848999, 495.5260925
2: -148.2908020, 304.5467834, -213.2291412, 423.0662231, -571.3570557, 517.7759399
3: -88.8953400, 367.7827759, -126.0885849, 515.1484985, -604.0437622, 493.8713684
4: -137.5424500, 263.9661865, -196.6598206, 366.8345642, -504.3770142, 460.6260071

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -39.1078987, 150.7899475, -60.0788307, 229.6140442, -268.7219238, 210.8687744
1: -104.1526108, 347.1518250, -160.7212830, 526.6140747, -630.7666626, 507.8731079
2: -148.2697906, 304.0660095, -235.5832825, 461.0300598, -609.2998657, 539.6492920
3: -88.7511673, 366.3943787, -137.7118073, 558.9442139, -647.6953125, 504.1062012
4: -137.4653473, 263.4257202, -216.5445862, 399.7543945, -537.2196655, 479.9703064

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -39.1078987, 150.7899475, -60.4919701, 231.4111023, -270.5190125, 211.2819214
1: -104.1526108, 347.1518250, -161.7693939, 531.5619507, -635.7145386, 508.9212036
2: -148.2697906, 304.0660095, -236.5246429, 464.6858826, -612.9556885, 540.5906372
3: -88.7511673, 366.3943787, -138.5607605, 564.0548706, -652.8060303, 504.9551392
4: -137.4653473, 263.4257202, -217.4815063, 402.9705200, -540.4358521, 480.9072266

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -39.1598473, 151.1481628, -60.0788307, 229.6140442, -268.7738953, 211.2269897
1: -104.3518448, 348.1903381, -160.7212830, 526.6140747, -630.9659424, 508.9116211
2: -148.2908020, 304.5467834, -235.5832825, 461.0300598, -609.3208618, 540.1300049
3: -88.8953400, 367.7827759, -137.7118073, 558.9442139, -647.8394775, 505.4945679
4: -137.5424500, 263.9661865, -216.5445862, 399.7543945, -537.2967529, 480.5107727

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -39.1598473, 151.1481628, -60.4919701, 231.4111023, -270.5709534, 211.6401367
1: -104.3518448, 348.1903381, -161.7693939, 531.5619507, -635.9138184, 509.9596863
2: -148.2908020, 304.5467834, -236.5246429, 464.6858826, -612.9766846, 541.0714111
3: -88.8953400, 367.7827759, -138.5607605, 564.0548706, -652.9501953, 506.3435364
4: -137.5424500, 263.9661865, -217.4815063, 402.9705200, -540.5129395, 481.4476929

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -44.9962158, 172.9530029, -54.7390404, 209.5062256, -254.5024414, 227.6920471
1: -119.4690475, 397.8204956, -146.5640564, 481.7807312, -601.2496338, 544.3845215
2: -172.2857513, 349.6840820, -212.6123810, 420.2774658, -592.5632324, 562.2964478
3: -102.0392227, 419.5346069, -125.4597855, 511.1200562, -613.1593018, 544.9943848
4: -159.1103210, 302.9974670, -196.0008545, 364.3806152, -523.4909058, 498.9983215

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7263120, upper bound: 187.7266868
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261828, upper bound: 187.7265820
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -44.9962158, 172.9530029, -55.0499115, 210.9465637, -255.9427643, 228.0029144
1: -119.4690475, 397.8204956, -147.3358307, 485.7331238, -605.2020874, 545.1563110
2: -172.2857513, 349.6840820, -213.2291412, 423.0662231, -595.3519897, 562.9132080
3: -102.0392227, 419.5346069, -126.0885849, 515.1484985, -617.1877441, 545.6231689
4: -159.1103210, 302.9974670, -196.6598206, 366.8345642, -525.9447632, 499.6572876

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7263120, upper bound: 187.7267851
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261828, upper bound: 187.7267057
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -45.3276634, 174.3791504, -54.7390404, 209.5062256, -254.8338776, 229.1181793
1: -120.3909531, 401.4574585, -146.5640564, 481.7807312, -602.1716919, 548.0214233
2: -173.2749176, 352.4047852, -212.6123810, 420.2774658, -593.5523682, 565.0171509
3: -102.7993240, 423.6196594, -125.4597855, 511.1200562, -613.9193726, 549.0792847
4: -160.1105652, 305.4512634, -196.0008545, 364.3806152, -524.4912109, 501.4521179

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259389, upper bound: 187.7261541
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261465, upper bound: 187.7265301
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -45.3276634, 174.3791504, -55.0499115, 210.9465637, -256.2742004, 229.4290466
1: -120.3909531, 401.4574585, -147.3358307, 485.7331238, -606.1240845, 548.7932739
2: -173.2749176, 352.4047852, -213.2291412, 423.0662231, -596.3411255, 565.6339111
3: -102.7993240, 423.6196594, -126.0885849, 515.1484985, -617.9477539, 549.7081909
4: -160.1105652, 305.4512634, -196.6598206, 366.8345642, -526.9450684, 502.1110840

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7259389, upper bound: 187.7261541
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7261464, upper bound: 187.7266074
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -44.9962158, 172.9530029, -60.0788307, 229.6140442, -274.6102600, 233.0318298
1: -119.4690475, 397.8204956, -160.7212830, 526.6140747, -646.0830688, 558.5417480
2: -172.2857513, 349.6840820, -235.5832825, 461.0300598, -633.3157959, 585.2673340
3: -102.0392227, 419.5346069, -137.7118073, 558.9442139, -660.9834595, 557.2463989
4: -159.1103210, 302.9974670, -216.5445862, 399.7543945, -558.8645630, 519.5420532

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -44.9962158, 172.9530029, -60.4919701, 231.4111023, -276.4073181, 233.4449768
1: -119.4690475, 397.8204956, -161.7693939, 531.5619507, -651.0309448, 559.5899048
2: -172.2857513, 349.6840820, -236.5246429, 464.6858826, -636.9716187, 586.2087402
3: -102.0392227, 419.5346069, -138.5607605, 564.0548706, -666.0941162, 558.0953369
4: -159.1103210, 302.9974670, -217.4815063, 402.9705200, -562.0807495, 520.4789429

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -45.3276634, 174.3791504, -60.0788307, 229.6140442, -274.9417114, 234.4579773
1: -120.3909531, 401.4574585, -160.7212830, 526.6140747, -647.0050049, 562.1787109
2: -173.2749176, 352.4047852, -235.5832825, 461.0300598, -634.3049927, 587.9879150
3: -102.7993240, 423.6196594, -137.7118073, 558.9442139, -661.7435303, 561.3314819
4: -160.1105652, 305.4512634, -216.5445862, 399.7543945, -559.8648071, 521.9958496

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -45.3276634, 174.3791504, -60.4919701, 231.4111023, -276.7387695, 234.8711090
1: -120.3909531, 401.4574585, -161.7693939, 531.5619507, -651.9528809, 563.2268066
2: -173.2749176, 352.4047852, -236.5246429, 464.6858826, -637.9608154, 588.9293823
3: -102.7993240, 423.6196594, -138.5607605, 564.0548706, -666.8541870, 562.1803589
4: -160.1105652, 305.4512634, -217.4815063, 402.9705200, -563.0810547, 522.9327393

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -52.0224991, 198.8103943, -39.9874573, 153.9800415, -206.0025330, 238.7978516
1: -139.0080414, 456.3614197, -105.6185684, 355.1460876, -494.1541138, 561.9799805
2: -201.8192444, 399.6740723, -149.6742096, 313.2406921, -515.0598755, 549.3482666
3: -119.0109406, 483.8226318, -89.9853439, 373.2077026, -492.2186279, 573.8079834
4: -186.0676880, 346.3368225, -139.0760345, 271.0276184, -457.0953064, 485.4128418

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -52.0224991, 198.8103943, -41.0267181, 158.1787720, -210.2012634, 239.8371124
1: -139.0080414, 456.3614197, -108.7297974, 365.2732544, -504.2813110, 565.0911865
2: -201.8192444, 399.6740723, -154.3438721, 320.4256287, -522.2448730, 554.0178223
3: -119.0109406, 483.8226318, -92.6975784, 384.6823120, -503.6932373, 576.5202026
4: -186.0676880, 346.3368225, -143.2359161, 277.5535278, -463.6212158, 489.5727234

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -51.6958923, 197.8632812, -41.5945625, 160.7980804, -212.4939728, 239.4578400
1: -138.4659271, 455.0554504, -110.1146317, 372.4962769, -510.9621887, 565.1699829
2: -200.8040314, 396.7898865, -155.0394135, 325.8557129, -526.6596069, 551.8292847
3: -118.5240784, 482.7778015, -93.7676773, 391.8578491, -510.3819275, 576.5454712
4: -185.1318970, 344.0062561, -144.2732697, 282.3645020, -467.4963989, 488.2795410

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7254834, upper bound: 187.7249053
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7254834, upper bound: 187.7249053
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -52.7015877, 201.6891327, -41.0510216, 158.2678528, -210.9694214, 242.7401581
1: -141.1636658, 463.7530212, -108.8065796, 365.4594421, -506.6231079, 572.5595703
2: -204.9100342, 404.3567200, -154.5256805, 320.5436096, -525.4536133, 558.8823853
3: -120.8423538, 492.0650635, -92.7737732, 384.9273987, -505.7697449, 584.8388672
4: -188.8331146, 350.5903320, -143.3928986, 277.6741028, -466.5072021, 493.9831848

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7258594, upper bound: 187.7252469
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7258594, upper bound: 187.7252540
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -51.2129440, 195.7597046, -41.7520676, 160.4431152, -211.6560516, 237.5117493
1: -136.7541504, 449.6931152, -111.8595047, 368.3892822, -505.1434326, 561.5526123
2: -198.4866486, 393.7602844, -162.0529938, 322.0512085, -520.5378418, 555.8132324
3: -117.0897827, 476.6947937, -95.4987869, 390.3433228, -507.4331055, 572.1936035
4: -183.0187836, 341.1972046, -149.2976074, 279.1828613, -462.2016602, 490.4948120

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252737, upper bound: 187.7246375
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -51.2129440, 195.7597046, -47.2421532, 181.1540527, -232.3669891, 243.0018616
1: -136.7541504, 449.6931152, -126.3357239, 415.2105713, -551.9647217, 576.0288086
2: -198.4866486, 393.7602844, -185.3231659, 363.8921509, -562.3787842, 579.0834351
3: -117.0897827, 476.6947937, -108.0975647, 440.1282959, -557.2180786, 584.7923584
4: -183.0187836, 341.1972046, -170.1908875, 315.5460815, -498.5648804, 511.3880920

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252737, upper bound: 187.7247151
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -52.3537979, 200.4330750, -43.6716614, 167.9504395, -220.3042297, 244.1047363
1: -140.0747986, 461.2239990, -117.0329285, 386.3254089, -526.4001465, 578.2569580
2: -203.0335999, 402.2585754, -169.1896362, 336.8154602, -539.8488770, 571.4482422
3: -119.9059677, 489.1560364, -99.9333954, 409.2757568, -529.1816406, 589.0894165
4: -187.1989746, 348.7276001, -155.9385681, 292.0995789, -479.2985229, 504.6661682

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252737, upper bound: 187.7251052
time: 1.39 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7256830, upper bound: 187.7250914
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -52.3537979, 200.4330750, -49.2578735, 188.9778595, -241.3316650, 249.6909485
1: -140.0747986, 461.2239990, -131.6718445, 433.9906311, -574.0653687, 592.8958740
2: -203.0335999, 402.2585754, -192.6070099, 379.6400757, -582.6737061, 594.8655396
3: -119.9059677, 489.1560364, -112.6752853, 459.9247437, -579.8306885, 601.8312988
4: -187.1989746, 348.7276001, -177.0028992, 329.2874451, -516.4864502, 525.7304688

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252737, upper bound: 187.7251135
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7254635, upper bound: 187.7254396
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7254635, upper bound: 187.7254396
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -57.3260422, 218.7723236, -42.5008125, 163.1286926, -220.4547424, 261.2731323
1: -153.0643768, 500.8387146, -112.6284103, 374.6515198, -527.7158813, 613.4671021
2: -224.7106781, 440.1973572, -162.6721039, 330.6642151, -555.3747559, 602.8694458
3: -131.1840057, 531.3409424, -96.1894073, 394.6486816, -525.8327026, 627.5303345
4: -206.5070190, 381.4566650, -150.1771545, 286.2350159, -492.7420349, 531.6337891

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -57.3260422, 218.7723236, -43.6014214, 167.5769043, -224.9029541, 262.3737488
1: -153.0643768, 500.8387146, -115.8522034, 385.5419006, -538.6062012, 616.6908569
2: -224.7106781, 440.1973572, -167.3560333, 338.3567505, -563.0673218, 607.5534058
3: -131.1840057, 531.3409424, -98.9713211, 406.8179932, -538.0020142, 630.3122559
4: -206.5070190, 381.4566650, -154.4309387, 293.2114258, -499.7184448, 535.8875732

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -58.1453362, 222.1918030, -42.5008125, 163.1286926, -221.2740173, 264.6926270
1: -155.6022644, 509.4779663, -112.6284103, 374.6515198, -530.2537842, 622.1063232
2: -228.3250580, 445.8673706, -162.6721039, 330.6642151, -558.9892578, 608.5394897
3: -133.3358917, 540.8165283, -96.1894073, 394.6486816, -527.9845581, 637.0058594
4: -209.7920380, 386.6254883, -150.1771545, 286.2350159, -496.0270386, 536.8024902

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -58.1453362, 222.1918030, -43.6020966, 167.5794067, -225.7247162, 265.7938843
1: -155.6022644, 509.4779663, -115.8541794, 385.5469360, -541.1491699, 625.3320923
2: -228.3250580, 445.8673706, -167.3597107, 338.3615417, -566.6865845, 613.2270508
3: -133.3358917, 540.8165283, -98.9729919, 406.8234863, -540.1593628, 639.7895508
4: -209.7920380, 386.6254883, -154.4340363, 293.2156067, -503.0076294, 541.0593872

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -56.5777512, 215.9529724, -41.7520676, 160.4431152, -217.0208588, 257.7050171
1: -150.9909363, 494.6424255, -111.8595047, 368.3892822, -519.3801880, 606.5018921
2: -221.6466675, 434.7843933, -162.0529938, 322.0512085, -543.6978149, 596.8374023
3: -129.4141846, 524.7028198, -95.4987869, 390.3433228, -519.7575073, 620.2015991
4: -203.6952209, 376.7477112, -149.2976074, 279.1828613, -482.8780823, 526.0452881

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7254526, upper bound: 187.7252150
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253028, upper bound: 187.7248701
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7255213, upper bound: 187.7255934
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -56.5777512, 215.9529724, -47.2421532, 181.1540527, -237.7317963, 263.1950684
1: -150.9909363, 494.6424255, -126.3357239, 415.2105713, -566.2015381, 620.9781494
2: -221.6466675, 434.7843933, -185.3231659, 363.8921509, -585.5387573, 620.1075439
3: -129.4141846, 524.7028198, -108.0975647, 440.1282959, -569.5424805, 632.8004150
4: -203.6952209, 376.7477112, -170.1908875, 315.5460815, -519.2412109, 546.9385986

Time for backsubstitution: 1.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252737, upper bound: 187.7252937
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253028, upper bound: 187.7248999
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7255213, upper bound: 187.7257088
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -57.7306824, 220.6721039, -43.6716614, 167.9504395, -225.6811218, 264.3437500
1: -154.3366547, 506.2919312, -117.0329285, 386.3254089, -540.6619873, 623.3248291
2: -226.1753998, 443.3265991, -169.1896362, 336.8154602, -562.9907227, 612.5162354
3: -132.2439117, 537.1520996, -99.9333954, 409.2757568, -541.5195312, 637.0853271
4: -207.8765717, 384.3719788, -155.9385681, 292.0995789, -499.9761353, 540.3104858

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7255321, upper bound: 187.7250750
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253388, upper bound: 187.7250304
time: 1.94 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7254932, upper bound: 187.7253258
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -57.7306824, 220.6721039, -49.2578735, 188.9778595, -246.7085419, 269.9299316
1: -154.3366547, 506.2919312, -131.6718445, 433.9906311, -588.3272705, 637.9637451
2: -226.1753998, 443.3265991, -192.6070099, 379.6400757, -605.8154907, 635.9335938
3: -132.2439117, 537.1520996, -112.6752853, 459.9247437, -592.1686401, 649.8272705
4: -207.8765717, 384.3719788, -177.0028992, 329.2874451, -537.1640015, 561.3748779

Time for backsubstitution: 1.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7255321, upper bound: 187.7250933
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253388, upper bound: 187.7250304
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7254932, upper bound: 187.7253497
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -52.8892097, 202.1773529, -52.4795151, 200.6813965, -253.5706024, 254.6568604
1: -141.1726074, 464.2333984, -140.3575745, 461.0521240, -602.2247314, 604.5907593
2: -204.5861511, 406.8985596, -203.9354553, 402.8723145, -607.4584961, 610.8339233
3: -120.8497620, 491.9433289, -120.1676941, 488.9883728, -609.8381348, 612.1110229
4: -188.7357483, 352.5541077, -187.9609985, 349.2334595, -537.9691772, 540.5150146

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -52.8892097, 202.1773529, -57.7324142, 220.4504395, -273.3395691, 259.9097595
1: -141.1726074, 464.2333984, -154.3140259, 505.1199646, -646.2926025, 618.5471802
2: -204.5861511, 406.8985596, -226.6507111, 442.8850403, -647.4711304, 633.5492554
3: -120.8497620, 491.9433289, -132.2461243, 536.0706787, -656.9204102, 624.1894531
4: -188.7357483, 352.5541077, -208.2572021, 383.9606934, -572.6963501, 560.8112793

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -53.7087860, 205.5821228, -54.0181389, 206.7471619, -260.4559326, 259.6002502
1: -143.7259827, 472.7937317, -144.5672913, 475.4723816, -619.1983643, 617.3610229
2: -208.2811432, 412.6273804, -209.5883331, 414.9142761, -623.1954346, 622.2156982
3: -123.0127182, 501.4205933, -123.7449188, 504.3094177, -627.3220825, 625.1655273
4: -192.0481720, 357.6968994, -193.2423248, 359.7003174, -551.7484741, 550.9392090

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252522, upper bound: 187.7253345
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252637, upper bound: 187.7253329
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -53.7087860, 205.5821228, -59.3369598, 226.7497101, -280.4584961, 264.9190674
1: -143.7259827, 472.7937317, -158.6584167, 520.0221558, -663.7481689, 631.4521484
2: -208.2811432, 412.6273804, -232.4867401, 455.5299988, -663.8111572, 645.1141357
3: -123.0127182, 501.4205933, -135.9401703, 551.7507324, -674.7634277, 637.3607788
4: -192.0481720, 357.6968994, -213.6977539, 394.9525757, -587.0007324, 571.3946533

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252522, upper bound: 187.7253534
time: 1.40 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7250473, upper bound: 187.7249401
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7250473, upper bound: 187.7252519
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -58.1915817, 222.1187592, -58.1674805, 222.0255280, -280.2171021, 280.2862549
1: -155.2480164, 508.5930481, -155.1849518, 508.4035950, -663.6515503, 663.7779541
2: -227.5268250, 447.4207153, -227.4449310, 447.1952820, -674.7221069, 674.8656006
3: -133.0295105, 539.2510376, -132.9751434, 539.0039062, -672.0334473, 672.2260742
4: -209.1822052, 387.6681213, -209.0946503, 387.4855957, -596.6677856, 596.7627563

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7250929, upper bound: 187.7248246
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7253137, upper bound: 187.7256286
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -58.1915817, 222.1187592, -58.9697952, 225.3521729, -283.5437317, 281.0885010
1: -155.2480164, 508.5930481, -157.6581116, 516.7785034, -672.0264893, 666.2511597
2: -227.5268250, 447.4207153, -230.9795074, 452.7807922, -680.3076172, 678.4002075
3: -133.0295105, 539.2510376, -135.0724335, 548.1978760, -681.2274170, 674.3233032
4: -209.1822052, 387.6681213, -212.2969208, 392.5490112, -601.7312012, 599.9649658

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7250929, upper bound: 187.7248246
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251544, upper bound: 187.7256286
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -59.0557251, 225.7017212, -54.0181389, 206.7471619, -265.8028870, 279.7198181
1: -157.8997955, 517.5921021, -144.5672913, 475.4723816, -633.3721924, 662.1594238
2: -231.2984924, 453.4600830, -209.5883331, 414.9142761, -646.2127686, 663.0483398
3: -135.2765198, 549.1116333, -123.7449188, 504.3094177, -639.5858765, 672.8565674
4: -212.6048279, 393.1467896, -193.2423248, 359.7003174, -572.3051147, 586.3890991

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251288, upper bound: 187.7249670
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -59.0557251, 225.7017212, -59.3369598, 226.7497101, -285.8054199, 285.0386658
1: -157.8997955, 517.5921021, -158.6584167, 520.0221558, -677.9219360, 676.2504883
2: -231.2984924, 453.4600830, -232.4867401, 455.5299988, -686.8284912, 685.9467773
3: -135.2765198, 549.1116333, -135.9401703, 551.7507324, -687.0272217, 685.0518188
4: -212.6048279, 393.1467896, -213.6977539, 394.9525757, -607.5573730, 606.8445435

Time for backsubstitution: 1.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7251288, upper bound: 187.7249670
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7252793, upper bound: 187.7252826
time: 1.04 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.85 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7270142, upper bound: 187.7270056
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7273414, upper bound: 187.7273414
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7270142, upper bound: 187.7270056
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7270142, upper bound: 187.7274779
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7270037, upper bound: 187.7269866
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7270059, upper bound: 187.7269889
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7270037, upper bound: 187.7269866
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7270059, upper bound: 187.7269889
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7258972, upper bound: 187.7257444
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7258972, upper bound: 187.7260036
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7258972, upper bound: 187.7258689
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7260036, upper bound: 187.7261290
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7272126, upper bound: 187.7274130
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7274130, upper bound: 187.7274471
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7274130, upper bound: 187.7274242
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7274129, upper bound: 187.7274598
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7261819, upper bound: 187.7263544
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7261819, upper bound: 187.7263544
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7259058, upper bound: 187.7257098
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7259058, upper bound: 187.7274767
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7263462, upper bound: 187.7260203
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7267927, upper bound: 187.7262270
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7262280, upper bound: 187.7259466
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7262427, upper bound: 187.7260482
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7249643, upper bound: 187.7247023
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7268853, upper bound: 187.7263198
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7245619, upper bound: 187.7245473
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7261437, upper bound: 187.7261437
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7260942, upper bound: 187.7264565
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7259520, upper bound: 187.7262334
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7260942, upper bound: 187.7264570
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7259520, upper bound: 187.7263012
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7263120, upper bound: 187.7266868
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7261828, upper bound: 187.7265820
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7263120, upper bound: 187.7267851
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7261828, upper bound: 187.7267057
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7259389, upper bound: 187.7261541
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7261465, upper bound: 187.7265301
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7259389, upper bound: 187.7261541
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7261464, upper bound: 187.7266074
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7254834, upper bound: 187.7249053
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7254834, upper bound: 187.7249053
NS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7258594, upper bound: 187.7252469
NS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7258594, upper bound: 187.7252540
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7252737, upper bound: 187.7251052
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7256830, upper bound: 187.7250914
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7254635, upper bound: 187.7254396
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7254635, upper bound: 187.7254396
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7253028, upper bound: 187.7248701
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7255213, upper bound: 187.7255934
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7253028, upper bound: 187.7248999
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7255213, upper bound: 187.7257088
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7253388, upper bound: 187.7250304
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7254932, upper bound: 187.7253258
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7253388, upper bound: 187.7250304
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7254932, upper bound: 187.7253497
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7252522, upper bound: 187.7253345
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7252637, upper bound: 187.7253329
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7250473, upper bound: 187.7249401
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7250473, upper bound: 187.7252519
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7250929, upper bound: 187.7248246
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7253137, upper bound: 187.7256286
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7250929, upper bound: 187.7248246
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7251544, upper bound: 187.7256286
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7251288, upper bound: 187.7249670
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 0, lower bound: -187.7252793, upper bound: 187.7252826

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -33.8257370, 129.9876709, -33.1969337, 127.5600891, -161.3858185, 163.1845856
1: -90.2916718, 299.0663147, -88.4948425, 293.3937683, -383.6854248, 387.5611572
2: -129.8908844, 261.4104919, -126.6191254, 256.9276733, -386.8185425, 388.0296021
3: -77.0358887, 316.7482605, -75.4675140, 310.6137085, -387.6495667, 392.2157593
4: -119.9472198, 226.6946106, -117.2082520, 222.7458038, -342.6930237, 343.9028625

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270142, upper bound: 187.7270002
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7269657, upper bound: 187.7269393
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -33.6022644, 129.0315399, -34.0349388, 130.7445374, -164.3468018, 163.0664520
1: -89.6930923, 296.5039062, -90.8267899, 300.4661560, -390.1592407, 387.3306885
2: -128.7354584, 259.3740234, -130.2100525, 262.9078979, -391.6433716, 389.5840759
3: -76.4881210, 314.0191956, -77.4499969, 318.1400452, -394.6281433, 391.4691467
4: -118.9811172, 224.8881226, -120.4078522, 227.9238739, -346.9049377, 345.2959595

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7271253, upper bound: 187.7271496
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270454, upper bound: 187.7270454
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -33.8257370, 129.9876709, -38.8834457, 149.0138550, -182.8395996, 168.8710938
1: -90.2916718, 299.0663147, -103.3708572, 342.0078125, -432.2994690, 402.4371643
2: -129.8908844, 261.4104919, -150.1046295, 300.9144287, -430.8052979, 411.5151367
3: -77.0358887, 316.7482605, -88.3519363, 361.8296509, -438.8655396, 405.1001892
4: -119.9472198, 226.6946106, -138.3305664, 260.8981628, -380.8453674, 365.0251770

Time for backsubstitution: 0.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272170, upper bound: 187.7271894
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270556, upper bound: 187.7270054
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -33.6022644, 129.0315399, -39.6587563, 151.9337006, -185.5359650, 168.6902924
1: -89.6930923, 296.5039062, -105.5413666, 348.3875427, -438.0806274, 402.0452881
2: -128.7354584, 259.3740234, -153.4870453, 306.5096436, -435.2451172, 412.8610840
3: -76.4881210, 314.0191956, -90.1925049, 368.7096558, -445.1977539, 404.2116699
4: -118.9811172, 224.8881226, -141.3310394, 265.6981506, -384.6792603, 366.2191772

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272170, upper bound: 187.7273299
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7271303, upper bound: 187.7271163
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -36.1798630, 139.6492462, -33.1969337, 127.5600891, -163.7399292, 172.8461761
1: -96.3615723, 321.6090698, -88.4948425, 293.3937683, -389.7553406, 410.1039124
2: -137.2158356, 281.6520996, -126.6191254, 256.9276733, -394.1434326, 408.2712097
3: -82.0646439, 339.4916992, -75.4675140, 310.6137085, -392.6782837, 414.9592285
4: -127.2201004, 244.1087341, -117.2082520, 222.7458038, -349.9659119, 361.3169556

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270037, upper bound: 187.7269823
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7269660, upper bound: 187.7269406
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -36.2286224, 139.6056671, -34.0349388, 130.7445374, -166.9731598, 173.6405487
1: -96.4505615, 321.3413086, -90.8267899, 300.4661560, -396.9167175, 412.1680908
2: -137.4152985, 281.6469421, -130.2100525, 262.9078979, -400.3231812, 411.8569946
3: -82.1519928, 338.9873047, -77.4499969, 318.1400452, -400.2920227, 416.4372253
4: -127.3588791, 243.9174194, -120.4078522, 227.9238739, -355.2827454, 364.3252563

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270059, upper bound: 187.7269845
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7269674, upper bound: 187.7269417
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -36.1798630, 139.6492462, -38.8834457, 149.0138550, -185.1937103, 178.5326843
1: -96.3615723, 321.6090698, -103.3708572, 342.0078125, -438.3693542, 424.9799194
2: -137.2158356, 281.6520996, -150.1046295, 300.9144287, -438.1302490, 431.7567139
3: -82.0646439, 339.4916992, -88.3519363, 361.8296509, -443.8942871, 427.8436279
4: -127.2201004, 244.1087341, -138.3305664, 260.8981628, -388.1182556, 382.4393005

Time for backsubstitution: 0.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272075, upper bound: 187.7271839
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270560, upper bound: 187.7270066
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -36.2286224, 139.6056671, -39.6587563, 151.9337006, -188.1623230, 179.2644196
1: -96.4505615, 321.3413086, -105.5413666, 348.3875427, -444.8381042, 426.8826904
2: -137.4152985, 281.6469421, -153.4870453, 306.5096436, -443.9249268, 435.1339722
3: -82.1519928, 338.9873047, -90.1925049, 368.7096558, -450.8616333, 429.1797791
4: -127.3588791, 243.9174194, -141.3310394, 265.6981506, -393.0570374, 385.2484741

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272097, upper bound: 187.7271862
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7270574, upper bound: 187.7270078
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -42.7218857, 164.8758392, -34.4278221, 132.9077148, -175.6295929, 199.3036499
1: -114.3567505, 380.5314636, -91.7263184, 306.2330627, -420.5898132, 472.2577820
2: -163.4930878, 331.1531677, -130.4829712, 267.9862061, -431.4792786, 461.6361389
3: -97.5029755, 402.5751953, -78.1068497, 322.9035950, -420.4065552, 480.6820374
4: -151.2417603, 287.2072754, -120.9742813, 232.0645294, -383.3062439, 408.1815491

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7245658, upper bound: 187.7256389
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7257565, upper bound: 187.7256372
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7255645, upper bound: 187.7255360
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -42.6336136, 164.0304108, -35.2350464, 136.0122986, -178.6459045, 199.2654572
1: -114.2265472, 377.2673645, -93.8776093, 313.4496460, -427.6761475, 471.1449280
2: -164.8068085, 329.1513062, -133.3950653, 274.2539062, -439.0607300, 462.5463562
3: -97.5026855, 399.5952759, -79.9408264, 330.5973816, -428.1000671, 479.5361023
4: -152.0125427, 285.4066772, -123.7474899, 237.4959259, -389.5084534, 409.1541748

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7260036, upper bound: 187.7260036
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7260036, upper bound: 187.7260036
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -42.7218857, 164.8758392, -40.2983932, 155.0016479, -197.7234955, 205.1742249
1: -114.3567505, 380.5314636, -107.0060730, 356.7510681, -471.1078186, 487.5375366
2: -163.4930878, 331.1531677, -154.6385651, 313.4049988, -476.8979492, 485.7917480
3: -97.5029755, 402.5751953, -91.3829956, 375.9282532, -473.4312134, 493.9581909
4: -151.2417603, 287.2072754, -142.6830750, 271.4890747, -422.7308350, 429.8903198

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7248226, upper bound: 187.7245752
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7249882, upper bound: 187.7247431
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -42.6336136, 164.0304108, -41.0951538, 158.0868683, -200.7204895, 205.1255646
1: -114.2265472, 377.2673645, -109.1275635, 363.9085999, -478.1351318, 486.3948669
2: -164.8068085, 329.1513062, -157.4803925, 319.6174927, -484.4243164, 486.6316833
3: -97.5026855, 399.5952759, -93.1899948, 383.5579834, -481.0606689, 492.7852783
4: -152.0125427, 285.4066772, -145.3733826, 276.8724670, -428.8849487, 430.7800598

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7249896, upper bound: 187.7249916
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7249882, upper bound: 187.7251683
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -37.9351768, 145.4536896, -38.2038651, 146.4945221, -184.4297028, 183.6575317
1: -100.9049988, 333.8540955, -101.6303711, 336.2579651, -437.1629639, 435.4844666
2: -146.5816956, 293.7179871, -147.6074219, 295.7599182, -442.3415833, 441.3254089
3: -86.2451935, 353.1665039, -86.8700714, 355.7000427, -441.9452515, 440.0365601
4: -135.0470734, 254.6722565, -135.9981079, 256.4407654, -391.4877930, 390.6703491

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272285, upper bound: 187.7272126
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272285, upper bound: 187.7274129
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -39.4060974, 151.1733398, -38.2038651, 146.4945221, -185.9006195, 189.3771820
1: -104.7422791, 347.1459045, -101.6303711, 336.2579651, -441.0002441, 448.7762146
2: -151.5406342, 305.0453491, -147.6074219, 295.7599182, -447.3005371, 452.6527710
3: -89.4565887, 366.8168640, -86.8700714, 355.7000427, -445.1565857, 453.6869202
4: -139.7576294, 264.3555603, -135.9981079, 256.4407654, -396.1983948, 400.3536682

Time for backsubstitution: 1.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272285, upper bound: 187.7272447
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -187.7272285, upper bound: 187.7274471
time: 0.89 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.91 + 418.17 = 421.08 seconds
