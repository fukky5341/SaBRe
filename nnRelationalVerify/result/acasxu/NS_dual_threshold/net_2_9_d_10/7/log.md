## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 4711.385590072957


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2886.2316895, 2318.2207031, -2886.2316895, 2318.2207031, -5204.4506836, 5204.4506836)
1: (-239.0890198, 170.8768463, -239.0890198, 170.8768463, -409.9658813, 409.9658813)
2: (-164.3258514, 276.7508240, -164.3258514, 276.7508240, -441.0766602, 441.0766602)
3: (-201.8939209, 408.3268433, -201.8939209, 408.3268433, -610.2207642, 610.2207642)
4: (-160.0726013, 280.1410217, -160.0726013, 280.1410217, -440.2135925, 440.2135925)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 3.01 + 1.90 = 4.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -4711.4327044, upper bound: 4711.4327043

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326986, upper bound: 4711.4326862
time: 0.51 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4327043, upper bound: 4711.4327043
time: 0.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.29 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.29
Output dim: 0, lower bound: -4711.4326986, upper bound: 4711.4326862
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.29
Output dim: 0, lower bound: -4711.4327043, upper bound: 4711.4327043

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2684.6606445, 2160.6528320, -2862.8247070, 2300.6127930, -4985.2734375, 5023.4775391
1: -222.8493195, 159.2016754, -237.2525940, 169.5366058, -392.3858337, 396.4542542
2: -153.5232697, 258.0295105, -163.0679321, 274.6791992, -428.2024536, 421.0974426
3: -188.4442749, 380.5668030, -200.3335266, 405.2163696, -593.6606445, 580.9002075
4: -149.5709839, 261.1437683, -158.8449249, 278.0166016, -427.5875549, 419.9886780

Time for backsubstitution: 2.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326823, upper bound: 4711.4326860
time: 0.56 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326987, upper bound: 4711.4326860
time: 0.63 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2882.1184082, 2315.4013672, -2886.2316895, 2318.2207031, -5200.3383789, 5201.6328125
1: -238.7884216, 170.6409760, -239.0890198, 170.8768463, -409.6652832, 409.7299805
2: -164.1049347, 276.4040527, -164.3258514, 276.7508240, -440.8557434, 440.7298889
3: -201.6273193, 407.7954712, -201.8939209, 408.3268433, -609.9541626, 609.6893921
4: -159.8612366, 279.7934875, -160.0726013, 280.1410217, -440.0021973, 439.8660278

Time for backsubstitution: 2.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326868
time: 0.64 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326789
time: 0.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.17 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.17
Output dim: 0, lower bound: -4711.4326823, upper bound: 4711.4326860
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.17
Output dim: 0, lower bound: -4711.4326987, upper bound: 4711.4326860
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.17
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326868
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.17
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326789

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -2608.0544434, 2098.9543457, -2600.7260742, 2115.1901855, -4723.2446289, 4699.6801758
1: -216.4350433, 154.7275085, -217.4515533, 154.4872131, -370.9222412, 372.1790771
2: -149.2850037, 251.0007019, -149.1206055, 252.3215332, -401.6065369, 400.1212463
3: -183.3456116, 370.0439758, -183.4272003, 371.0108643, -554.3564453, 553.4711304
4: -145.6765900, 253.8062286, -145.4280090, 255.6229401, -401.2994080, 399.2341919

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326823, upper bound: 4711.4326649
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326823, upper bound: 4711.4326858
time: 0.62 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2684.6606445, 2160.6528320, -2856.0964355, 2295.7629395, -4980.4238281, 5016.7490234
1: -222.8493195, 159.2016754, -236.7557068, 169.1315460, -391.9808350, 395.9573669
2: -153.5232697, 258.0295105, -162.7363281, 274.0472412, -427.5704956, 420.7658386
3: -188.4442749, 380.5668030, -199.9288177, 404.2852783, -592.7295532, 580.4953613
4: -149.5709839, 261.1437683, -158.4732513, 277.4112854, -426.9822693, 419.6170044

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326988, upper bound: 4711.4326651
time: 0.52 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326988, upper bound: 4711.4326859
time: 0.60 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -2866.6987305, 2305.5307617, -2886.2316895, 2318.2207031, -5184.9184570, 5191.7607422
1: -237.7119751, 169.7759552, -239.0890198, 170.8768463, -408.5888062, 408.8649902
2: -163.2950287, 275.1787415, -164.3258514, 276.7508240, -440.0458374, 439.5045776
3: -200.6579590, 405.8450623, -201.8939209, 408.3268433, -608.9848022, 607.7390137
4: -159.0712433, 278.5867920, -160.0726013, 280.1410217, -439.2122498, 438.6593628

Time for backsubstitution: 2.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326651, upper bound: 4711.4326688
time: 0.64 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326869
time: 0.69 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -2940.0444336, 2348.4230957, -2878.4946289, 2312.9582520, -5253.0019531, 5226.9169922
1: -242.4223633, 173.8162994, -238.5279541, 170.4344482, -412.8568115, 412.3442383
2: -166.6899261, 280.9093628, -163.9241791, 276.0994263, -442.7893677, 444.8335266
3: -204.8841095, 414.8082581, -201.4111633, 407.3223877, -612.2063599, 616.2193604
4: -162.5794983, 283.9697266, -159.6826782, 279.5028992, -442.0823975, 443.6524048

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326655, upper bound: 4711.4326391
time: 0.56 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326789
time: 0.64 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.23 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.23
Output dim: 0, lower bound: -4711.4326823, upper bound: 4711.4326649
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.23
Output dim: 0, lower bound: -4711.4326823, upper bound: 4711.4326858
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.23
Output dim: 0, lower bound: -4711.4326988, upper bound: 4711.4326651
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.23
Output dim: 0, lower bound: -4711.4326988, upper bound: 4711.4326859
NS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.23
Output dim: 0, lower bound: -4711.4326651, upper bound: 4711.4326688
NS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.23
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326869
NS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.23
Output dim: 0, lower bound: -4711.4326655, upper bound: 4711.4326391
NS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.23
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326789

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2403.7863770, 1965.1978760, -2600.7260742, 2115.1901855, -4518.9765625, 4565.9238281
1: -201.9353333, 143.0654602, -217.4515533, 154.4872131, -356.4225464, 360.5169678
2: -138.7184753, 234.3088989, -149.1206055, 252.3215332, -391.0400085, 383.4294128
3: -170.5514069, 344.1552124, -183.4272003, 371.0108643, -541.5622559, 527.5823975
4: -135.2913055, 237.4986877, -145.4280090, 255.6229401, -390.9142151, 382.9266968

Time for backsubstitution: 2.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326624, upper bound: 4711.4326651
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326822, upper bound: 4711.4326644
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2679.7026367, 2156.8117676, -2600.7260742, 2115.1901855, -4794.8925781, 4757.5380859
1: -222.4656219, 158.8932190, -217.4515533, 154.4872131, -376.9527893, 376.3447876
2: -153.2691956, 257.5426025, -149.1206055, 252.3215332, -405.5907288, 406.6632080
3: -188.1300049, 379.8563232, -183.4272003, 371.0108643, -559.1408691, 563.2833252
4: -149.2776031, 260.6744995, -145.4280090, 255.6229401, -404.9005432, 406.1024780

Time for backsubstitution: 2.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326810, upper bound: 4711.4326857
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326821, upper bound: 4711.4326858
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2403.7863770, 1965.1978760, -2856.0964355, 2295.7629395, -4699.5493164, 4821.2944336
1: -201.9353333, 143.0654602, -236.7557068, 169.1315460, -371.0668640, 379.8211060
2: -138.7184753, 234.3088989, -162.7363281, 274.0472412, -412.7657166, 397.0451965
3: -170.5514069, 344.1552124, -199.9288177, 404.2852783, -574.8366699, 544.0839233
4: -135.2913055, 237.4986877, -158.4732513, 277.4112854, -412.7025757, 395.9719238

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326623, upper bound: 4711.4326647
time: 0.51 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326822, upper bound: 4711.4326650
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2679.7026367, 2156.8117676, -2856.0964355, 2295.7629395, -4975.4658203, 5012.9082031
1: -222.4656219, 158.8932190, -236.7557068, 169.1315460, -391.5970764, 395.6488953
2: -153.2691956, 257.5426025, -162.7363281, 274.0472412, -427.3164368, 420.2789307
3: -188.1300049, 379.8563232, -199.9288177, 404.2852783, -592.4152222, 579.7849121
4: -149.2776031, 260.6744995, -158.4732513, 277.4112854, -426.6889038, 419.1477356

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326805, upper bound: 4711.4326855
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326818, upper bound: 4711.4326860
time: 0.63 seconds

## BFS NS instance: NS_A2_A1_A1

### Backsubstitution after applying NS history:
0: -2849.0041504, 2292.6745605, -2883.0751953, 2315.9165039, -5164.9189453, 5175.7490234
1: -236.2194366, 168.9178925, -238.8427277, 170.6981812, -406.9176025, 407.7606201
2: -162.3559113, 273.6845093, -164.1576691, 276.4694214, -438.8253174, 437.8421021
3: -199.2728119, 403.3065796, -201.6807556, 407.8995056, -607.1723022, 604.9872437
4: -157.9245453, 277.0313721, -159.8976898, 279.8631287, -437.7876587, 436.9290466

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_A1_A1_B1

### Relational analysis result of NS_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326348, upper bound: 4711.4326655
time: 0.64 seconds

## Relational analysis of NS_A2_A1_A1_B2

### Relational analysis result of NS_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326651, upper bound: 4711.4326687
time: 0.66 seconds

## BFS NS instance: NS_A2_A1_A2

### Backsubstitution after applying NS history:
0: -2830.3088379, 2274.0234375, -2886.2316895, 2318.2207031, -5148.5283203, 5160.2529297
1: -234.5069885, 167.5665436, -239.0890198, 170.8768463, -405.3838196, 406.6555786
2: -161.1104584, 271.5520325, -164.3258514, 276.7508240, -437.8612671, 435.8778687
3: -197.9982758, 400.5339050, -201.8939209, 408.3268433, -606.3251343, 602.4278564
4: -156.9939117, 274.7887878, -160.0726013, 280.1410217, -437.1349487, 434.8613586

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_A1_A2_A1

### Relational analysis result of NS_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326784, upper bound: 4711.4326571
time: 0.62 seconds

## Relational analysis of NS_A2_A1_A2_A2

### Relational analysis result of NS_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326869
time: 0.59 seconds

## BFS NS instance: NS_A2_A2_A1

### Backsubstitution after applying NS history:
0: -2913.7329102, 2330.5456543, -2875.3391113, 2310.6582031, -5224.3911133, 5205.8842773
1: -240.3436127, 172.4985809, -238.2818909, 170.2558136, -410.5994263, 410.7804565
2: -165.4115295, 278.7654114, -163.7559967, 275.8183289, -441.2298279, 442.5214233
3: -202.9748383, 411.3586426, -201.1980743, 406.8952637, -609.8700562, 612.5566406
4: -160.9997406, 281.7238770, -159.5077972, 279.2253418, -440.2250977, 441.2316589

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_A2_A1_B1

### Relational analysis result of NS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326348, upper bound: 4711.4326385
time: 0.68 seconds

## Relational analysis of NS_A2_A2_A1_B2

### Relational analysis result of NS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326656, upper bound: 4711.4326388
time: 0.65 seconds

## BFS NS instance: NS_A2_A2_A2

### Backsubstitution after applying NS history:
0: -2904.8144531, 2317.3671875, -2878.4946289, 2312.9582520, -5217.7724609, 5195.8608398
1: -239.2649841, 171.6822205, -238.5279541, 170.4344482, -409.6994019, 410.2101746
2: -164.5279236, 277.3016968, -163.9241791, 276.0994263, -440.6273499, 441.2258606
3: -202.2049713, 409.6155396, -201.4111633, 407.3223877, -609.5270996, 611.0265503
4: -160.5391541, 280.1997070, -159.6826782, 279.5028992, -440.0420532, 439.8823547

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_A2_A2_B1

### Relational analysis result of NS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326383, upper bound: 4711.4326783
time: 0.57 seconds

## Relational analysis of NS_A2_A2_A2_B2

### Relational analysis result of NS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326788
time: 0.58 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.27 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326624, upper bound: 4711.4326651
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326822, upper bound: 4711.4326644
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326810, upper bound: 4711.4326857
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326821, upper bound: 4711.4326858
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326623, upper bound: 4711.4326647
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326822, upper bound: 4711.4326650
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326805, upper bound: 4711.4326855
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326818, upper bound: 4711.4326860
NS_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326348, upper bound: 4711.4326655
NS_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326651, upper bound: 4711.4326687
NS_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326784, upper bound: 4711.4326571
NS_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326869
NS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326348, upper bound: 4711.4326385
NS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326656, upper bound: 4711.4326388
NS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326383, upper bound: 4711.4326783
NS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.27
Output dim: 0, lower bound: -4711.4326789, upper bound: 4711.4326788

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2401.7807617, 1963.3090820, -2582.3103027, 2094.2309570, -4496.0107422, 4545.6191406
1: -201.7459106, 142.9378662, -215.4196472, 153.3484344, -355.0942993, 358.3575134
2: -138.5911255, 234.0915985, -147.8474426, 250.1778717, -388.7689819, 381.9390259
3: -170.3953857, 343.8479309, -181.7592010, 368.0523376, -538.4476929, 525.6071167
4: -135.1701355, 237.2757874, -144.1777802, 253.1664734, -388.3366089, 381.4535522

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326623, upper bound: 4711.4326644
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326624, upper bound: 4711.4326651
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2403.7863770, 1965.1978760, -2575.1574707, 2093.4240723, -4497.2104492, 4540.3554688
1: -201.9353333, 143.0654602, -215.2425842, 152.9584045, -354.8937378, 358.3080444
2: -138.7184753, 234.3088989, -147.6468353, 249.8197632, -388.5382385, 381.9556580
3: -170.5514069, 344.1552124, -181.6238251, 367.3716431, -537.9230347, 525.7790527
4: -135.2913055, 237.4986877, -144.0216217, 252.9817657, -388.2730103, 381.5202637

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326820, upper bound: 4711.4326644
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326822, upper bound: 4711.4326651
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -2627.6940918, 2112.7141113, -2577.1281738, 2096.5432129, -4724.2368164, 4689.8422852
1: -217.9252319, 155.7380676, -215.5249481, 153.1109772, -371.0361023, 371.2630005
2: -150.1607666, 252.2379913, -147.8161926, 250.1439667, -400.3046875, 400.0541992
3: -184.3648376, 372.0599060, -181.8224335, 367.7833862, -552.1481934, 553.8823242
4: -146.3803864, 255.4309845, -144.1777802, 253.3706818, -399.7510681, 399.6087646

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326778, upper bound: 4711.4326792
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326778, upper bound: 4711.4326792
time: 0.57 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -2671.7612305, 2150.6550293, -2600.7260742, 2115.1901855, -4786.9511719, 4751.3793945
1: -221.8263245, 158.4282379, -217.4515533, 154.4872131, -376.3135376, 375.8797913
2: -152.8205566, 256.8033752, -149.1206055, 252.3215332, -405.1420898, 405.9238892
3: -187.5850525, 378.7664185, -183.4272003, 371.0108643, -558.5958862, 562.1935425
4: -148.8458405, 259.9272461, -145.4280090, 255.6229401, -404.4687195, 405.3552551

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326787, upper bound: 4711.4326834
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326806, upper bound: 4711.4326834
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2401.7807617, 1963.3090820, -2847.6059570, 2280.1381836, -4681.9184570, 4810.9150391
1: -201.7459106, 142.9378662, -235.3087006, 168.5501404, -370.2960510, 378.2465515
2: -138.5911255, 234.0915985, -161.9543304, 272.6261902, -411.2173157, 396.0458984
3: -170.3953857, 343.8479309, -198.8350677, 402.5005798, -572.8959961, 542.6829834
4: -135.1701355, 237.2757874, -157.6916504, 275.6106567, -410.7807922, 394.9674377

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326759, upper bound: 4711.4326644
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326756, upper bound: 4711.4326649
time: 0.53 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2403.7863770, 1965.1978760, -2829.1809082, 2273.7973633, -4677.5834961, 4794.3784180
1: -201.9353333, 143.0654602, -234.5131378, 167.5420685, -369.4773865, 377.5784912
2: -138.7184753, 234.3088989, -161.2135315, 271.4933777, -410.2118225, 395.5223389
3: -170.5514069, 344.1552124, -198.0637665, 400.5311890, -571.0825806, 542.2188721
4: -135.2913055, 237.4986877, -157.0175018, 274.7356873, -410.0269775, 394.5161743

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326985, upper bound: 4711.4326644
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326986, upper bound: 4711.4326649
time: 0.56 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -2627.6940918, 2112.7141113, -2831.6738281, 2275.7868652, -4903.4809570, 4944.3876953
1: -217.9252319, 155.7380676, -234.7062531, 167.6983337, -385.6235352, 390.4443359
2: -150.1607666, 252.2379913, -161.3568420, 271.7374573, -421.8982239, 413.5948486
3: -184.3648376, 372.0599060, -198.2254181, 400.8995056, -585.2643433, 570.2853394
4: -146.3803864, 255.4309845, -157.1592865, 275.0111694, -421.3915405, 412.5902100

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326922, upper bound: 4711.4326790
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326932, upper bound: 4711.4326791
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -2671.7612305, 2150.6550293, -2856.0964355, 2295.7629395, -4967.5244141, 5006.7500000
1: -221.8263245, 158.4282379, -236.7557068, 169.1315460, -390.9578552, 395.1839600
2: -152.8205566, 256.8033752, -162.7363281, 274.0472412, -426.8677979, 419.5396423
3: -187.5850525, 378.7664185, -199.9288177, 404.2852783, -591.8702393, 578.6951294
4: -148.8458405, 259.9272461, -158.4732513, 277.4112854, -426.2571411, 418.4005127

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326970, upper bound: 4711.4326833
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326986, upper bound: 4711.4326834
time: 0.70 seconds

## BFS NS instance: NS_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -2776.1977539, 2231.9841309, -2620.6018066, 2130.2497559, -4906.4472656, 4852.5854492
1: -229.9432678, 164.6119843, -219.0177917, 155.6268616, -385.5701294, 383.6297302
2: -158.1812897, 266.8130493, -150.1654205, 254.0833893, -412.2646790, 416.9783936
3: -194.2619171, 393.1326294, -184.7251892, 373.6495361, -567.9114380, 577.8577881
4: -154.1610718, 269.8084412, -146.4489288, 257.4428101, -411.6038818, 416.2573242

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_A1_A1_B1_A1

### Relational analysis result of NS_A2_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326345, upper bound: 4711.4326476
time: 0.64 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2

### Relational analysis result of NS_A2_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326346, upper bound: 4711.4326662
time: 0.64 seconds

## BFS NS instance: NS_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -2849.0041504, 2292.6745605, -2876.3637695, 2311.0739746, -5160.0771484, 5169.0380859
1: -236.2194366, 168.9178925, -238.3467255, 170.2939453, -406.5133667, 407.2646179
2: -162.3559113, 273.6845093, -163.8267517, 275.8383484, -438.1942749, 437.5112305
3: -199.2728119, 403.3065796, -201.2765503, 406.9697571, -606.2425537, 604.5830688
4: -157.9245453, 277.0313721, -159.5266571, 279.2587585, -437.1832886, 436.5580139

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_A1_A1_B2_A1

### Relational analysis result of NS_A2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326637, upper bound: 4711.4326493
time: 0.63 seconds

## Relational analysis of NS_A2_A1_A1_B2_A2

### Relational analysis result of NS_A2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326638, upper bound: 4711.4326687
time: 0.76 seconds

## BFS NS instance: NS_A2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -2561.6579590, 2085.1306152, -2811.0461426, 2256.4509277, -4818.1083984, 4896.1752930
1: -214.3257904, 152.1506958, -232.6807251, 166.4419403, -380.7677307, 384.8313904
2: -146.8546753, 248.7323761, -160.0867920, 269.7341919, -416.5888672, 408.8191528
3: -180.7482758, 365.5763855, -196.8004456, 397.8941956, -578.6424561, 562.3767700
4: -143.2579956, 251.9668579, -156.2131958, 272.7853699, -416.0433655, 408.1800537

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_A1_A2_A1_B1

### Relational analysis result of NS_A2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326557
time: 0.58 seconds

## Relational analysis of NS_A2_A1_A2_A1_B2

### Relational analysis result of NS_A2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326571
time: 0.78 seconds

## BFS NS instance: NS_A2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -2823.8562012, 2269.3881836, -2886.2316895, 2318.2207031, -5142.0756836, 5155.6196289
1: -234.0321045, 167.1779327, -239.0890198, 170.8768463, -404.9089050, 406.2669678
2: -160.7932587, 270.9452209, -164.3258514, 276.7508240, -437.5440063, 435.2710571
3: -197.6112671, 399.6394653, -201.8939209, 408.3268433, -605.9381104, 601.5333862
4: -156.6363220, 274.2088318, -160.0726013, 280.1410217, -436.7773132, 434.2814026

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_A1_A2_A2_B1

### Relational analysis result of NS_A2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326850
time: 0.72 seconds

## Relational analysis of NS_A2_A1_A2_A2_B2

### Relational analysis result of NS_A2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326868
time: 0.58 seconds

## BFS NS instance: NS_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -2841.1284180, 2271.7714844, -2613.0822754, 2125.1271973, -4966.2558594, 4884.8535156
1: -234.1810150, 168.1818237, -218.4715271, 155.1932373, -389.3741760, 386.6532593
2: -161.3029175, 271.9970398, -149.7740326, 253.4470367, -414.7499390, 421.7710571
3: -198.0675049, 401.3328247, -184.2546387, 372.6687927, -570.7362671, 585.5874634
4: -157.2160797, 274.6677551, -146.0657349, 256.8207703, -414.0368652, 420.7334595

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_A2_A1_B1_A1

### Relational analysis result of NS_A2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326346, upper bound: 4711.4326131
time: 0.54 seconds

## Relational analysis of NS_A2_A2_A1_B1_A2

### Relational analysis result of NS_A2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326384
time: 0.55 seconds

## BFS NS instance: NS_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -2913.7329102, 2330.5456543, -2868.6318359, 2305.8210449, -5219.5537109, 5199.1772461
1: -240.3436127, 172.4985809, -237.7864990, 169.8518524, -410.1954346, 410.2850647
2: -165.4115295, 278.7654114, -163.4253693, 275.1877441, -440.5992737, 442.1907654
3: -202.9748383, 411.3586426, -200.7942810, 405.9661255, -608.9409790, 612.1529541
4: -160.9997406, 281.7238770, -159.1369019, 278.6215515, -439.6212769, 440.8607788

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_A2_A1_B2_A1

### Relational analysis result of NS_A2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326638, upper bound: 4711.4326131
time: 0.57 seconds

## Relational analysis of NS_A2_A2_A1_B2_A2

### Relational analysis result of NS_A2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326639, upper bound: 4711.4326389
time: 0.56 seconds

## BFS NS instance: NS_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -2831.7951660, 2259.2873535, -2616.2397461, 2127.4675293, -4959.2622070, 4875.5258789
1: -233.1674652, 167.3433533, -218.7209778, 155.3722534, -388.5396118, 386.0643311
2: -160.4818115, 270.6013794, -149.9463806, 253.7311096, -414.2128906, 420.5477600
3: -197.3777008, 399.5889282, -184.4745483, 373.0993347, -570.4770508, 584.0634766
4: -156.7616119, 273.2592163, -146.2355652, 257.1024475, -413.8640747, 419.4947815

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326383
time: 0.59 seconds

## Relational analysis of NS_A2_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326783
time: 0.57 seconds

## BFS NS instance: NS_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2904.8144531, 2317.3671875, -2871.7888184, 2308.1232910, -5212.9375000, 5189.1562500
1: -239.2649841, 171.6822205, -238.0326385, 170.0305328, -409.2955017, 409.7148132
2: -164.5279236, 277.3016968, -163.5936127, 275.4689941, -439.9969177, 440.8953247
3: -202.2049713, 409.6155396, -201.0074768, 406.3933411, -608.5982666, 610.6229858
4: -160.5391541, 280.1997070, -159.3118744, 278.8992310, -439.4383850, 439.5115356

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326783, upper bound: 4711.4326383
time: 0.80 seconds

## Relational analysis of NS_A2_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326784, upper bound: 4711.4326789
time: 0.70 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.69 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326623, upper bound: 4711.4326644
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326624, upper bound: 4711.4326651
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326820, upper bound: 4711.4326644
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326822, upper bound: 4711.4326651
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326778, upper bound: 4711.4326792
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326778, upper bound: 4711.4326792
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326787, upper bound: 4711.4326834
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326806, upper bound: 4711.4326834
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326759, upper bound: 4711.4326644
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326756, upper bound: 4711.4326649
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326985, upper bound: 4711.4326644
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326986, upper bound: 4711.4326649
NS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326922, upper bound: 4711.4326790
NS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326932, upper bound: 4711.4326791
NS_A1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326970, upper bound: 4711.4326833
NS_A1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326986, upper bound: 4711.4326834
NS_A2_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326345, upper bound: 4711.4326476
NS_A2_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326346, upper bound: 4711.4326662
NS_A2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326637, upper bound: 4711.4326493
NS_A2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326638, upper bound: 4711.4326687
NS_A2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326557
NS_A2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326571
NS_A2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326850
NS_A2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326868
NS_A2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326346, upper bound: 4711.4326131
NS_A2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326384
NS_A2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326638, upper bound: 4711.4326131
NS_A2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326639, upper bound: 4711.4326389
NS_A2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326383
NS_A2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326783
NS_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326783, upper bound: 4711.4326383
NS_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 0, lower bound: -4711.4326784, upper bound: 4711.4326789

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2333.6093750, 1907.4630127, -2559.9826660, 2076.2810059, -4409.8906250, 4467.4458008
1: -196.0671539, 138.8078003, -213.5774994, 152.0346985, -348.1018677, 352.3853149
2: -134.7108612, 227.5685425, -146.5814209, 248.0876923, -382.7985229, 374.1499634
3: -165.6702576, 334.0908203, -180.2201538, 364.9695435, -530.6397705, 514.3109741
4: -131.5486145, 230.5709686, -142.9811554, 251.0038452, -382.5524597, 373.5521240

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326590, upper bound: 4711.4326630
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326623, upper bound: 4711.4326641
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2395.4265137, 1958.4532471, -2582.3103027, 2094.2309570, -4489.6562500, 4540.7636719
1: -201.2401581, 142.5673370, -215.4196472, 153.3484344, -354.5885315, 357.9869995
2: -138.2333679, 233.5050201, -147.8474426, 250.1778717, -388.4112549, 381.3524780
3: -169.9613342, 342.9793396, -181.7592010, 368.0523376, -538.0135498, 524.7385254
4: -134.8226929, 236.6854553, -144.1777802, 253.1664734, -387.9891663, 380.8632202

Time for backsubstitution: 2.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326623, upper bound: 4711.4326649
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326581, upper bound: 4711.4326633
time: 1.03 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 21

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2335.3256836, 1909.0961914, -2551.4201660, 2074.8969727, -4410.2226562, 4460.5166016
1: -196.2299957, 138.9186249, -213.3252258, 151.5839996, -347.8139648, 352.2438354
2: -134.8207092, 227.7548370, -146.3341064, 247.6466675, -382.4673767, 374.0889282
3: -165.8054657, 334.3545532, -180.0087128, 364.1387634, -529.9442139, 514.3632202
4: -131.6523285, 230.7650909, -142.7648315, 250.7510834, -382.4033508, 373.5298767

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326756, upper bound: 4711.4326580
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326684, upper bound: 4711.4326579
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2397.4060059, 1960.3205566, -2575.1574707, 2093.4240723, -4490.8300781, 4535.4775391
1: -201.4274139, 142.6934204, -215.2425842, 152.9584045, -354.3858032, 357.9360046
2: -138.3591919, 233.7197723, -147.6468353, 249.8197632, -388.1789551, 381.3665466
3: -170.1155548, 343.2828674, -181.6238251, 367.3716431, -537.4871826, 524.9066772
4: -134.9423981, 236.9057007, -144.0216217, 252.9817657, -387.9240723, 380.9272156

Time for backsubstitution: 2.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326787, upper bound: 4711.4326621
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326803, upper bound: 4711.4326618
time: 0.52 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -2600.6435547, 2090.1169434, -2404.4228516, 1953.3223877, -4553.9658203, 4494.5400391
1: -215.5849609, 154.1590271, -200.7161102, 142.9785004, -358.5634766, 354.8750916
2: -148.6027985, 249.6235352, -137.9590302, 233.8058319, -382.4086304, 387.5825806
3: -182.4496613, 368.2382812, -169.5862427, 344.0048218, -526.4543457, 537.8245239
4: -144.8653107, 252.6827850, -134.6033020, 236.0703125, -380.9355469, 387.2860107

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A1_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326648, upper bound: 4711.4326706
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326648, upper bound: 4711.4326706
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -2627.6940918, 2112.7141113, -2557.1306152, 2082.4804688, -4710.1743164, 4669.8447266
1: -217.9252319, 155.7380676, -213.9990692, 151.9752502, -369.9004211, 369.7371216
2: -150.1607666, 252.2379913, -146.7418213, 248.3242493, -398.4850159, 398.9797974
3: -184.3648376, 372.0599060, -180.4904633, 365.0336914, -549.3984985, 552.5503540
4: -146.3803864, 255.4309845, -143.0689697, 251.6606293, -398.0410156, 398.4999390

Time for backsubstitution: 2.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326767, upper bound: 4711.4326791
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326675, upper bound: 4711.4326790
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -2645.4799805, 2128.2590332, -2427.0805664, 1971.0562744, -4616.5361328, 4555.3398438
1: -219.5338440, 156.8777618, -202.5484009, 144.2953796, -363.8291931, 359.4261475
2: -151.2800598, 254.2632751, -139.2091675, 235.9015961, -387.1815796, 393.4724426
3: -185.7003937, 375.0239868, -171.1276398, 347.0869751, -532.7871704, 546.1515503
4: -147.3558350, 257.2166748, -135.8110199, 238.2167816, -385.5726318, 393.0277100

Time for backsubstitution: 2.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326655, upper bound: 4711.4326725
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326649, upper bound: 4711.4326724
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2671.7612305, 2150.6550293, -2580.6875000, 2101.0996094, -4772.8608398, 4731.3427734
1: -221.8263245, 158.4282379, -215.9211884, 153.3482056, -375.1745300, 374.3494263
2: -152.8205566, 256.8033752, -148.0632019, 250.4941864, -403.3147278, 404.8665161
3: -187.5850525, 378.7664185, -182.1197052, 368.2525635, -555.8375244, 560.8861084
4: -148.8458405, 259.9272461, -144.3150177, 253.9060822, -402.7518616, 404.2422180

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326665, upper bound: 4711.4326724
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326620, upper bound: 4711.4326724
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2333.6093750, 1907.4630127, -2824.0212402, 2260.6186523, -4594.2280273, 4731.4843750
1: -196.0671539, 138.8078003, -233.3137970, 167.1607361, -363.2279053, 372.1215820
2: -134.7108612, 227.5685425, -160.5982361, 270.3731995, -405.0840454, 388.1667786
3: -165.6702576, 334.0908203, -197.1613617, 399.2266235, -564.8967896, 531.2521973
4: -131.5486145, 230.5709686, -156.4223480, 273.2708130, -404.8193665, 386.9933167

Time for backsubstitution: 2.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326736, upper bound: 4711.4326630
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326710, upper bound: 4711.4326600
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2395.4265137, 1958.4532471, -2847.6059570, 2280.1381836, -4675.5634766, 4806.0590820
1: -201.2401581, 142.5673370, -235.3087006, 168.5501404, -369.7902832, 377.8760376
2: -138.2333679, 233.5050201, -161.9543304, 272.6261902, -410.8595581, 395.4593506
3: -169.9613342, 342.9793396, -198.8350677, 402.5005798, -572.4619141, 541.8143921
4: -134.8226929, 236.6854553, -157.6916504, 275.6106567, -410.4333496, 394.3771057

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326708, upper bound: 4711.4326608
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326726, upper bound: 4711.4326633
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 17

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2335.3256836, 1909.0961914, -2804.7822266, 2253.9509277, -4589.2763672, 4713.8784180
1: -196.2299957, 138.9186249, -232.4768524, 166.1200104, -362.3500061, 371.3954773
2: -134.8207092, 227.7548370, -159.8359222, 269.1954956, -404.0162048, 387.5907593
3: -165.8054657, 334.3545532, -196.3663330, 397.1579895, -562.9634399, 530.7207642
4: -131.6523285, 230.7650909, -155.7069702, 272.3591309, -404.0114746, 386.4719849

Time for backsubstitution: 2.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326980, upper bound: 4711.4326630
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326922, upper bound: 4711.4326631
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2397.4060059, 1960.3205566, -2829.1809082, 2273.7973633, -4671.2031250, 4789.5000000
1: -201.4274139, 142.6934204, -234.5131378, 167.5420685, -368.9694519, 377.2065125
2: -138.3591919, 233.7197723, -161.2135315, 271.4933777, -409.8525391, 394.9332275
3: -170.1155548, 343.2828674, -198.0637665, 400.5311890, -570.6467285, 541.3466187
4: -134.9423981, 236.9057007, -157.0175018, 274.7356873, -409.6780396, 393.9231873

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326965, upper bound: 4711.4326623
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326987, upper bound: 4711.4326620
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -2600.6435547, 2090.1169434, -2654.9238281, 2131.7573242, -4732.4003906, 4745.0410156
1: -215.5849609, 154.1590271, -219.7523499, 157.3719482, -372.9569092, 373.9113770
2: -148.6027985, 249.6235352, -151.3790436, 255.2025146, -403.8052979, 401.0025635
3: -182.4496613, 368.2382812, -185.8793488, 376.7196045, -559.1692505, 554.1175537
4: -144.8653107, 252.6827850, -147.4631805, 257.5735779, -402.4388123, 400.1459656

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326828, upper bound: 4711.4326705
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326648, upper bound: 4711.4326702
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -2627.6940918, 2112.7141113, -2813.7009277, 2262.8759766, -4890.5703125, 4926.4150391
1: -217.9252319, 155.7380676, -233.3145599, 166.6796722, -384.6048584, 389.0526123
2: -150.1607666, 252.2379913, -160.4019928, 270.0859680, -420.2467041, 412.6399841
3: -184.3648376, 372.0599060, -197.0530548, 398.4188232, -582.7836914, 569.1129761
4: -146.3803864, 255.4309845, -156.1486664, 273.4405212, -419.8209229, 411.5796509

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326933, upper bound: 4711.4326791
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326677, upper bound: 4711.4326790
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -2645.4799805, 2128.2590332, -2678.6320801, 2150.9902344, -4796.4702148, 4806.8911133
1: -219.5338440, 156.8777618, -221.7308197, 158.7589722, -378.2928162, 378.6085815
2: -151.2800598, 254.2632751, -152.7255096, 257.4599609, -408.7400208, 406.9887695
3: -185.7003937, 375.0239868, -187.5395966, 379.9925537, -565.6929321, 562.5634766
4: -147.3558350, 257.2166748, -148.7475891, 259.8950195, -407.2508545, 405.9641724

Time for backsubstitution: 2.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326831, upper bound: 4711.4326742
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326831, upper bound: 4711.4326724
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2671.7612305, 2150.6550293, -2838.0769043, 2282.8703613, -4954.6318359, 4988.7319336
1: -221.8263245, 158.4282379, -235.3643646, 168.1115417, -389.9378662, 393.7926025
2: -152.8205566, 256.8033752, -161.7982635, 272.3926392, -425.2131042, 418.6016235
3: -187.5850525, 378.7664185, -198.7775116, 401.8013611, -589.3862915, 577.5438843
4: -148.8458405, 259.9272461, -157.4616852, 275.8406982, -424.6865234, 417.3889160

Time for backsubstitution: 2.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326786, upper bound: 4711.4326567
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326984, upper bound: 4711.4326831
time: 0.73 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2580.6391602, 2102.8737793, -2620.6018066, 2130.2497559, -4710.8886719, 4723.4750977
1: -215.9633331, 153.4947205, -219.0177917, 155.6268616, -371.5902100, 372.5125122
2: -148.0331879, 250.8036957, -150.1654205, 254.0833893, -402.1165771, 400.9690552
3: -181.9310303, 368.2841187, -184.7251892, 373.6495361, -555.5803833, 553.0092163
4: -144.1849518, 254.1132355, -146.4489288, 257.4428101, -401.6276855, 400.5620728

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_A1_B1_A1_B1

### Relational analysis result of NS_A2_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326345, upper bound: 4711.4326476
time: 0.74 seconds

## Relational analysis of NS_A2_A1_A1_B1_A1_B2

### Relational analysis result of NS_A2_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326345, upper bound: 4711.4326476
time: 0.57 seconds

## BFS NS instance: NS_A2_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2842.5429688, 2288.0429688, -2620.6018066, 2130.2497559, -4972.7924805, 4908.6445312
1: -235.7453766, 168.5289917, -219.0177917, 155.6268616, -391.3722534, 387.5467834
2: -162.0387115, 273.0823669, -150.1654205, 254.0833893, -416.1221008, 423.2477112
3: -198.8851624, 402.4166565, -184.7251892, 373.6495361, -572.5346069, 587.1417236
4: -157.5710144, 276.4527283, -146.4489288, 257.4428101, -415.0138245, 422.9016113

Time for backsubstitution: 2.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_A1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_A1_B1_A2_B1

### Relational analysis result of NS_A2_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326346, upper bound: 4711.4326660
time: 0.68 seconds

## Relational analysis of NS_A2_A1_A1_B1_A2_B2

### Relational analysis result of NS_A2_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326345, upper bound: 4711.4326662
time: 0.67 seconds

## BFS NS instance: NS_A2_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2580.6391602, 2102.8737793, -2876.3637695, 2311.0739746, -4891.7128906, 4979.2368164
1: -215.9633331, 153.4947205, -238.3467255, 170.2939453, -386.2572632, 391.8414307
2: -148.0331879, 250.8036957, -163.8267517, 275.8383484, -423.8715210, 414.6304321
3: -181.9310303, 368.2841187, -201.2765503, 406.9697571, -588.9007568, 569.5606079
4: -144.1849518, 254.1132355, -159.5266571, 279.2587585, -423.4436951, 413.6398621

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_A1_B2_A1_B1

### Relational analysis result of NS_A2_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326346, upper bound: 4711.4326493
time: 0.59 seconds

## Relational analysis of NS_A2_A1_A1_B2_A1_B2

### Relational analysis result of NS_A2_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326346, upper bound: 4711.4326493
time: 0.53 seconds

## BFS NS instance: NS_A2_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2842.5429688, 2288.0429688, -2876.3637695, 2311.0739746, -5153.6166992, 5164.4067383
1: -235.7453766, 168.5289917, -238.3467255, 170.2939453, -406.0393066, 406.8757324
2: -162.0387115, 273.0823669, -163.8267517, 275.8383484, -437.8770752, 436.9091187
3: -198.8851624, 402.4166565, -201.2765503, 406.9697571, -605.8549194, 603.6931763
4: -157.5710144, 276.4527283, -159.5266571, 279.2587585, -436.8297729, 435.9793701

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 48

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_A1_B2_A2_B1

### Relational analysis result of NS_A2_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326345, upper bound: 4711.4326674
time: 0.69 seconds

## Relational analysis of NS_A2_A1_A1_B2_A2_B2

### Relational analysis result of NS_A2_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326345, upper bound: 4711.4326674
time: 0.55 seconds

## BFS NS instance: NS_A2_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -2561.6579590, 2085.1306152, -2623.7502441, 2132.5849609, -4694.2412109, 4708.8808594
1: -214.3257904, 152.1506958, -219.2666626, 155.8054047, -370.1311951, 371.4173584
2: -146.8546753, 248.7323761, -150.3381958, 254.3667603, -401.2214355, 399.0705566
3: -180.7482758, 365.5763855, -184.9446259, 374.0790405, -554.8273315, 550.5209351
4: -143.2579956, 251.9668579, -146.6183472, 257.7238159, -400.9818115, 398.5851135

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326383, upper bound: 4711.4326528
time: 0.60 seconds

## Relational analysis of NS_A2_A1_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326557
time: 0.70 seconds

## BFS NS instance: NS_A2_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -2561.6579590, 2085.1306152, -2879.5212402, 2313.3791504, -4875.0361328, 4964.6499023
1: -214.3257904, 152.1506958, -238.5931244, 170.4726868, -384.7984619, 390.7437439
2: -146.8546753, 248.7323761, -163.9950256, 276.1199036, -422.9745789, 412.7274170
3: -180.7482758, 365.5763855, -201.4898682, 407.3973694, -588.1456299, 567.0662231
4: -143.2579956, 251.9668579, -159.7016754, 279.5367737, -422.7947693, 411.6685181

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 17

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 18

## Relational analysis of NS_A2_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326571
time: 0.60 seconds

## Relational analysis of NS_A2_A1_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326571
time: 0.60 seconds

## BFS NS instance: NS_A2_A1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -2823.8562012, 2269.3881836, -2623.7502441, 2132.5849609, -4956.4399414, 4893.1386719
1: -234.0321045, 167.1779327, -219.2666626, 155.8054047, -389.8374634, 386.4445801
2: -160.7932587, 270.9452209, -150.3381958, 254.3667603, -415.1598816, 421.2833557
3: -197.6112671, 399.6394653, -184.9446259, 374.0790405, -571.6903076, 584.5841064
4: -156.6363220, 274.2088318, -146.6183472, 257.7238159, -414.3601379, 420.8271179

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_A1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_A2_A2_B1_B1

### Relational analysis result of NS_A2_A1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326383, upper bound: 4711.4326852
time: 0.70 seconds

## Relational analysis of NS_A2_A1_A2_A2_B1_B2

### Relational analysis result of NS_A2_A1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326383, upper bound: 4711.4326852
time: 0.68 seconds

## BFS NS instance: NS_A2_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2823.8562012, 2269.3881836, -2879.5212402, 2313.3791504, -5137.2348633, 5148.9082031
1: -234.0321045, 167.1779327, -238.5931244, 170.4726868, -404.5047302, 405.7710571
2: -160.7932587, 270.9452209, -163.9950256, 276.1199036, -436.9131165, 434.9402466
3: -197.6112671, 399.6394653, -201.4898682, 407.3973694, -605.0086670, 601.1293335
4: -156.6363220, 274.2088318, -159.7016754, 279.5367737, -436.1730957, 433.9104919

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_A2_A2_B2_B1

### Relational analysis result of NS_A2_A1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326863
time: 0.69 seconds

## Relational analysis of NS_A2_A1_A2_A2_B2_B2

### Relational analysis result of NS_A2_A1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326864
time: 0.72 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2657.5878906, 2150.9348145, -2613.0822754, 2125.1271973, -4782.7148438, 4764.0170898
1: -221.1648254, 157.8137054, -218.4715271, 155.1932373, -376.3580322, 376.2852173
2: -151.7617950, 257.0923767, -149.7740326, 253.4470367, -405.2088013, 406.8663940
3: -186.5467072, 378.1108093, -184.2546387, 372.6687927, -559.2153931, 562.3654175
4: -147.8911896, 260.0504150, -146.0657349, 256.8207703, -404.7119751, 406.1161499

Time for backsubstitution: 3.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_A1_B1_A1_B1

### Relational analysis result of NS_A2_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326347, upper bound: 4711.4326130
time: 0.57 seconds

## Relational analysis of NS_A2_A2_A1_B1_A1_B2

### Relational analysis result of NS_A2_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326347, upper bound: 4711.4326131
time: 0.56 seconds

## BFS NS instance: NS_A2_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2907.4892578, 2326.1325684, -2613.0822754, 2125.1271973, -5032.6162109, 4939.2148438
1: -239.8933105, 172.1246185, -218.4715271, 155.1932373, -395.0864258, 390.5961304
2: -165.1063538, 278.1880493, -149.7740326, 253.4470367, -418.5534058, 427.9620972
3: -202.6067810, 410.5037231, -184.2546387, 372.6687927, -575.2755127, 594.7583618
4: -160.6613617, 281.1735535, -146.0657349, 256.8207703, -417.4821167, 427.2392578

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_A1_B1_A2_B1

### Relational analysis result of NS_A2_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326347, upper bound: 4711.4326385
time: 0.62 seconds

## Relational analysis of NS_A2_A2_A1_B1_A2_B2

### Relational analysis result of NS_A2_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326347, upper bound: 4711.4326384
time: 0.73 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2657.5878906, 2150.9348145, -2868.6318359, 2305.8210449, -4963.4091797, 5019.5664062
1: -221.1648254, 157.8137054, -237.7864990, 169.8518524, -391.0166626, 395.6001587
2: -151.7617950, 257.0923767, -163.4253693, 275.1877441, -426.9495239, 420.5177612
3: -186.5467072, 378.1108093, -200.7942810, 405.9661255, -592.5128174, 578.9050903
4: -147.8911896, 260.0504150, -159.1369019, 278.6215515, -426.5127563, 419.1873169

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 17

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_A1_B2_A1_B1

### Relational analysis result of NS_A2_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326347, upper bound: 4711.4326131
time: 0.56 seconds

## Relational analysis of NS_A2_A2_A1_B2_A1_B2

### Relational analysis result of NS_A2_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326347, upper bound: 4711.4326131
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2907.4892578, 2326.1325684, -2868.6318359, 2305.8210449, -5213.3100586, 5194.7641602
1: -239.8933105, 172.1246185, -237.7864990, 169.8518524, -409.7450562, 409.9111328
2: -165.1063538, 278.1880493, -163.4253693, 275.1877441, -440.2940979, 441.6134033
3: -202.6067810, 410.5037231, -200.7942810, 405.9661255, -608.5728760, 611.2979736
4: -160.6613617, 281.1735535, -159.1369019, 278.6215515, -439.2828674, 440.3104553

Time for backsubstitution: 2.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_A1_B2_A2_B1

### Relational analysis result of NS_A2_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326389
time: 0.68 seconds

## Relational analysis of NS_A2_A2_A1_B2_A2_B2

### Relational analysis result of NS_A2_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326347, upper bound: 4711.4326389
time: 0.66 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2649.1525879, 2140.0710449, -2616.2397461, 2127.4675293, -4776.6196289, 4756.3095703
1: -220.2735138, 157.0390778, -218.7209778, 155.3722534, -375.6457520, 375.7600403
2: -151.0123138, 255.8171234, -149.9463806, 253.7311096, -404.7434082, 405.7634888
3: -185.9351349, 376.5614624, -184.4745483, 373.0993347, -559.0344849, 561.0358887
4: -147.4902344, 258.7446289, -146.2355652, 257.1024475, -404.5926819, 404.9801331

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_A2_B1_A1_A1

### Relational analysis result of NS_A2_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326368
time: 0.68 seconds

## Relational analysis of NS_A2_A2_A2_B1_A1_A2

### Relational analysis result of NS_A2_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326384
time: 0.56 seconds

## BFS NS instance: NS_A2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2898.0466309, 2312.6193848, -2616.2397461, 2127.4675293, -5025.5141602, 4928.8588867
1: -238.7756805, 171.2780914, -218.7209778, 155.3722534, -394.1477966, 389.9990234
2: -164.1959076, 276.6784363, -149.9463806, 253.7311096, -417.9270020, 426.6248169
3: -201.8049316, 408.6900024, -184.4745483, 373.0993347, -574.9042969, 593.1645508
4: -160.1695862, 279.6077576, -146.2355652, 257.1024475, -417.2720337, 425.8433228

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_A2_B1_A2_A1

### Relational analysis result of NS_A2_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326701
time: 0.72 seconds

## Relational analysis of NS_A2_A2_A2_B1_A2_A2

### Relational analysis result of NS_A2_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326383, upper bound: 4711.4326784
time: 0.59 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2649.1525879, 2140.0710449, -2871.7888184, 2308.1232910, -4957.2758789, 5011.8598633
1: -220.2735138, 157.0390778, -238.0326385, 170.0305328, -390.3040466, 395.0716248
2: -151.0123138, 255.8171234, -163.5936127, 275.4689941, -426.4813232, 419.4107361
3: -185.9351349, 376.5614624, -201.0074768, 406.3933411, -592.3284912, 577.5689087
4: -147.4902344, 258.7446289, -159.3118744, 278.8992310, -426.3894348, 418.0564270

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_A2_B2_A1_A1

### Relational analysis result of NS_A2_A2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326369
time: 0.58 seconds

## Relational analysis of NS_A2_A2_A2_B2_A1_A2

### Relational analysis result of NS_A2_A2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326383, upper bound: 4711.4326384
time: 0.65 seconds

## BFS NS instance: NS_A2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2898.0466309, 2312.6193848, -2871.7888184, 2308.1232910, -5206.1699219, 5184.4082031
1: -238.7756805, 171.2780914, -238.0326385, 170.0305328, -408.8061829, 409.3106384
2: -164.1959076, 276.6784363, -163.5936127, 275.4689941, -439.6649170, 440.2720337
3: -201.8049316, 408.6900024, -201.0074768, 406.3933411, -608.1982422, 609.6975098
4: -160.1695862, 279.6077576, -159.3118744, 278.8992310, -439.0687866, 438.9196167

Time for backsubstitution: 2.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 17

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_A2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_A2_A2_B2_A2_A1

### Relational analysis result of NS_A2_A2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326704
time: 0.59 seconds

## Relational analysis of NS_A2_A2_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326788
time: 0.58 seconds

## Relational analysis of NS_A2_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326789
time: 0.78 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 7.13 seconds
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326756, upper bound: 4711.4326580
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326684, upper bound: 4711.4326579
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326787, upper bound: 4711.4326621
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326803, upper bound: 4711.4326618
NS_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326648, upper bound: 4711.4326706
NS_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326648, upper bound: 4711.4326706
NS_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326767, upper bound: 4711.4326791
NS_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326675, upper bound: 4711.4326790
NS_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326655, upper bound: 4711.4326725
NS_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326649, upper bound: 4711.4326724
NS_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326665, upper bound: 4711.4326724
NS_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326620, upper bound: 4711.4326724
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326980, upper bound: 4711.4326630
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326922, upper bound: 4711.4326631
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326965, upper bound: 4711.4326623
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326987, upper bound: 4711.4326620
NS_A1_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326828, upper bound: 4711.4326705
NS_A1_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326648, upper bound: 4711.4326702
NS_A1_B2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326933, upper bound: 4711.4326791
NS_A1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326677, upper bound: 4711.4326790
NS_A1_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326831, upper bound: 4711.4326742
NS_A1_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326831, upper bound: 4711.4326724
NS_A1_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326786, upper bound: 4711.4326567
NS_A1_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326984, upper bound: 4711.4326831
NS_A2_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326345, upper bound: 4711.4326476
NS_A2_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326345, upper bound: 4711.4326476
NS_A2_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326346, upper bound: 4711.4326660
NS_A2_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326345, upper bound: 4711.4326662
NS_A2_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326346, upper bound: 4711.4326493
NS_A2_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326346, upper bound: 4711.4326493
NS_A2_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326345, upper bound: 4711.4326674
NS_A2_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326345, upper bound: 4711.4326674
NS_A2_A1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326383, upper bound: 4711.4326528
NS_A2_A1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326557
NS_A2_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326571
NS_A2_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326571
NS_A2_A1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326383, upper bound: 4711.4326852
NS_A2_A1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326383, upper bound: 4711.4326852
NS_A2_A1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326863
NS_A2_A1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326864
NS_A2_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326347, upper bound: 4711.4326130
NS_A2_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326347, upper bound: 4711.4326131
NS_A2_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326347, upper bound: 4711.4326385
NS_A2_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326347, upper bound: 4711.4326384
NS_A2_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326347, upper bound: 4711.4326131
NS_A2_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326347, upper bound: 4711.4326131
NS_A2_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326344, upper bound: 4711.4326389
NS_A2_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326347, upper bound: 4711.4326389
NS_A2_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326368
NS_A2_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326384
NS_A2_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326701
NS_A2_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326383, upper bound: 4711.4326784
NS_A2_A2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326369
NS_A2_A2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326383, upper bound: 4711.4326384
NS_A2_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326788
NS_A2_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.13
Output dim: 0, lower bound: -4711.4326384, upper bound: 4711.4326789

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2335.3256836, 1909.0961914, -2425.8991699, 1981.2475586, -4316.5727539, 4334.9951172
1: -196.2299957, 138.9186249, -203.4892120, 144.2450867, -340.4750977, 342.4078064
2: -134.8207092, 227.7548370, -139.3428040, 235.9083252, -370.7290039, 367.0976562
3: -165.8054657, 334.3545532, -171.4088287, 346.5554504, -512.3609009, 505.7633667
4: -131.6523285, 230.7650909, -135.7810974, 239.3626404, -371.0149536, 366.5461121

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326680, upper bound: 4711.4326579
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326680, upper bound: 4711.4326580
time: 0.55 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2335.3256836, 1909.0961914, -2465.3647461, 1995.3177490, -4330.6430664, 4374.4609375
1: -196.2299957, 138.9186249, -205.3014832, 146.2297821, -342.4597473, 344.2200623
2: -134.8207092, 227.7548370, -140.8448334, 238.4676056, -373.2883301, 368.5996704
3: -165.8054657, 334.3545532, -173.3303680, 350.8113098, -516.6167603, 507.6849060
4: -131.6523285, 230.7650909, -137.4270935, 241.1921844, -372.8445129, 368.1921082

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326684, upper bound: 4711.4326577
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326684, upper bound: 4711.4326580
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2373.2644043, 1939.1748047, -2408.0761719, 1955.2264404, -4328.4907227, 4347.2509766
1: -199.2671356, 141.2688446, -200.9284210, 143.1783905, -342.4454956, 342.1972656
2: -136.9250488, 231.3455353, -138.1282043, 234.0580597, -370.9830933, 369.4737549
3: -168.3574982, 339.8106079, -169.7733917, 344.4238586, -512.7813721, 509.5839844
4: -133.5552979, 234.3438568, -134.7772827, 236.3128204, -369.8681030, 369.1211548

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326653, upper bound: 4711.4326570
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326648, upper bound: 4711.4326417
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2397.4060059, 1960.3205566, -2554.9741211, 2079.0566406, -4476.4628906, 4515.2949219
1: -201.4274139, 142.6934204, -213.6761932, 151.7922516, -353.2196655, 356.3695984
2: -138.3591919, 233.7197723, -146.5323029, 247.9702606, -386.3294678, 380.2520447
3: -170.1155548, 343.2828674, -180.2456512, 364.5795593, -534.6951294, 523.5285034
4: -134.9423981, 236.9057007, -142.8977051, 251.2366791, -386.1789551, 379.8034058

Time for backsubstitution: 2.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326652, upper bound: 4711.4326570
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326616, upper bound: 4711.4326417
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -2600.4931641, 2090.0261230, -2270.1633301, 1847.9948730, -4448.4877930, 4360.1894531
1: -215.5748901, 154.1505737, -189.8132019, 135.0109863, -350.5857849, 343.9637451
2: -148.5948029, 249.6119385, -130.3461761, 220.9402771, -369.5350952, 379.9581299
3: -182.4403076, 368.2203064, -160.2519226, 324.6647034, -507.1050110, 528.4722290
4: -144.8577271, 252.6714630, -127.1269226, 223.2905426, -368.1482544, 379.7983398

Time for backsubstitution: 2.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326647, upper bound: 4711.4326705
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326647, upper bound: 4711.4326707
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -2600.6435547, 2090.1169434, -2335.6071777, 1890.2178955, -4490.8613281, 4425.7241211
1: -215.5849609, 154.1590271, -194.2456360, 138.7208405, -354.3057861, 348.4046631
2: -148.6027985, 249.6235352, -133.7820740, 226.4195557, -375.0223389, 383.4056091
3: -182.4496613, 368.2382812, -164.3936462, 333.4371338, -515.8867188, 532.6319580
4: -144.8653107, 252.6827850, -130.5309753, 228.2887268, -373.1540222, 383.2137451

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326645, upper bound: 4711.4326705
time: 0.54 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326647, upper bound: 4711.4326706
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -2627.6940918, 2112.7141113, -2542.7976074, 2070.9323730, -4698.6264648, 4655.5117188
1: -217.9252319, 155.7380676, -212.8123016, 151.1038818, -369.0291138, 368.5503540
2: -150.1607666, 252.2379913, -145.8930969, 246.8884583, -397.0491943, 398.1311035
3: -184.3648376, 372.0599060, -179.4678345, 362.9512634, -547.3159790, 551.5277100
4: -146.3803864, 255.4309845, -142.2246094, 250.2519836, -396.6323853, 397.6555786

Time for backsubstitution: 2.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326645, upper bound: 4711.4326706
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326599, upper bound: 4711.4326706
time: 0.54 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -2623.7136230, 2109.2429199, -2559.2788086, 2082.0100098, -4705.7221680, 4668.5214844
1: -217.5713348, 155.5046387, -213.9718781, 152.0847015, -369.6560364, 369.4765015
2: -149.9211426, 251.8300629, -146.8005829, 248.3286743, -398.2498169, 398.6306458
3: -184.0681305, 371.4860840, -180.5298462, 365.1524353, -549.2205811, 552.0158691
4: -146.1454773, 255.0057831, -143.1074371, 251.5580902, -397.7035522, 398.1132202

Time for backsubstitution: 2.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 25

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326635, upper bound: 4711.4326706
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326589, upper bound: 4711.4326706
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -2645.4799805, 2128.2590332, -2292.8354492, 1865.5941162, -4511.0742188, 4421.0947266
1: -219.5338440, 156.8777618, -191.6455383, 136.3325958, -355.8664246, 348.5232849
2: -151.2800598, 254.2632751, -131.6237488, 223.0251007, -374.3051453, 385.8870239
3: -185.7003937, 375.0239868, -161.8088989, 327.7461243, -513.4465332, 536.8328857
4: -147.3558350, 257.2166748, -128.3528290, 225.4131317, -372.7689209, 385.5694580

Time for backsubstitution: 2.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: B, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 45

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_A2_B1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326649, upper bound: 4711.4326724
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_B1_A2

### Relational analysis result of NS_A1_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -4711.4326649, upper bound: 4711.4326724
time: 0.71 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.91 + 415.77 = 420.68 seconds
