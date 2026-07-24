## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_1.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 112.5390001482


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448)
1: (-24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693)
2: (-20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918)
3: (-40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481)
4: (-30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820)

## BASE Result
execution time: IAR + LP analysis = 2.15 + 1.74 = 3.90 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -112.6516518, upper bound: 112.6516518


# Binary Search by BASE starts (time budget: 1196.10 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0625000, mid=0.0625000, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072838]}

## Binary search (step 1) starts
Candidate diff: 0.0312500


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0312500, mid=0.0312500, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072838, 112.65165179072841]}

## Binary search (step 2) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0156250, mid=0.0156250, abs_max=129.51934814453125
rel_dist={3: [-112.65131788498067, 112.65131788498067]}

## Binary search (step 3) starts
Candidate diff: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, low=0.0000000, high=0.0078125, mid=0.0078125, abs_max=129.51934814453125
rel_dist={3: [-112.65078213774841, 112.65078213774842]}

## Binary search (step 4) starts
Candidate diff: 0.0039062


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0000000, high=0.0039062, mid=0.0039062, abs_max=129.51934814453125
rel_dist={3: [-112.6504189427153, 112.6504189427153]}

## Binary search (step 5) starts
Candidate diff: 0.0019531


## IAR start
Binary search (step 5): status=Status.UNKNOWN, low=0.0000000, high=0.0019531, mid=0.0019531, abs_max=129.51934814453125
rel_dist={3: [-112.65017967261763, 112.65017967261764]}

## Binary search (step 6) starts
Candidate diff: 0.0009766


## IAR start
Binary search (step 6): status=Status.UNKNOWN, low=0.0000000, high=0.0009766, mid=0.0009766, abs_max=129.51934814453125
rel_dist={3: [-112.6500328643804, 112.65003286438042]}

## Binary search (step 7) starts
Candidate diff: 0.0004883


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0000000, high=0.0004883, mid=0.0004883, abs_max=129.51934814453125
rel_dist={3: [-112.64995858793105, 112.64995858793105]}

## Binary search (step 8) starts
Candidate diff: 0.0002441


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0000000, high=0.0002441, mid=0.0002441, abs_max=129.51934814453125
rel_dist={3: [-112.64992144971538, 112.64992144971535]}

## Binary search (step 9) starts
Candidate diff: 0.0001221


## IAR start
Binary search (step 9): status=Status.UNKNOWN, low=0.0000000, high=0.0001221, mid=0.0001221, abs_max=129.51934814453125
rel_dist={3: [-112.6499028806254, 112.6499028806254]}

## Binary search (step 10) starts
Candidate diff: 0.0000610


## IAR start
Binary search (step 10): status=Status.UNKNOWN, low=0.0000000, high=0.0000610, mid=0.0000610, abs_max=129.51934814453125
rel_dist={3: [-112.64989359611607, 112.64989359611604]}

## Binary search (step 11) starts
Candidate diff: 0.0000305


## IAR start
Binary search (step 11): status=Status.UNKNOWN, low=0.0000000, high=0.0000305, mid=0.0000305, abs_max=129.51934814453125
rel_dist={3: [-112.64988895393194, 112.64988895393194]}

## Binary search (step 12) starts
Candidate diff: 0.0000153


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0000000, high=0.0000153, mid=0.0000153, abs_max=129.51934814453125
rel_dist={3: [-112.64988663297817, 112.64988663297817]}

## Binary search (step 13) starts
Candidate diff: 0.0000076


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0000000, high=0.0000076, mid=0.0000076, abs_max=129.51934814453125
rel_dist={3: [-112.64988547276707, 112.64988547276707]}

## Binary search (step 14) starts
Candidate diff: 0.0000038


## IAR start
Binary search (step 14): status=Status.UNKNOWN, low=0.0000000, high=0.0000038, mid=0.0000038, abs_max=129.51934814453125
rel_dist={3: [-112.64988489315357, 112.64988488756634]}

## Binary search (step 15) starts
Candidate diff: 0.0000019


## IAR start
Binary search (step 15): status=Status.UNKNOWN, low=0.0000000, high=0.0000019, mid=0.0000019, abs_max=129.51934814453125
rel_dist={3: [-112.6498846482485, 112.64988459666108]}

## Binary search (step 16) starts
Candidate diff: 0.0000010


## IAR start
Binary search (step 16): status=Status.UNKNOWN, low=0.0000000, high=0.0000010, mid=0.0000010, abs_max=129.51934814453125
rel_dist={3: [-112.6498847525292, 112.64988487598032]}

## Binary Search Result
Binary search time: 66.16 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual) starts
Time budget: 1129.95 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6367131, upper bound: 112.6214954
time: 0.68 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6513812, upper bound: 112.6513812
time: 0.82 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.69 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 3, lower bound: -112.6367131, upper bound: 112.6214954
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.69
Output dim: 3, lower bound: -112.6513812, upper bound: 112.6513812

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.6501446, 77.5402985, -14.8509598, 83.8702927, -97.5204315, 92.3912582
1: -22.1772804, 92.1125412, -24.1805515, 99.6739273, -121.8512115, 116.2930908
2: -19.2295589, 92.0258636, -20.9075127, 99.5991058, -118.8286591, 112.9333572
3: -37.4879189, 81.5224380, -40.9821777, 88.5371628, -126.0250549, 122.5046082
4: -28.0775452, 80.3824844, -30.6739483, 87.2144394, -115.2919769, 111.0564270

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5277605, upper bound: 112.5693728
time: 0.59 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4706063, upper bound: 112.4484261
time: 0.63 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -14.7116480, 83.1687164, -14.8509598, 83.8702927, -98.5819321, 98.0196686
1: -23.9496403, 98.8321838, -24.1805515, 99.6739273, -123.6235657, 123.0127258
2: -20.7152519, 98.7536392, -20.9075127, 99.5991058, -120.3143311, 119.6611328
3: -40.5850754, 87.7603989, -40.9821777, 88.5371628, -129.1222229, 128.7425537
4: -30.3844452, 86.4514465, -30.6739483, 87.2144394, -117.5988846, 117.1253967

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5986886, upper bound: 112.5421197
time: 0.81 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
time: 0.85 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.82 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.82
Output dim: 3, lower bound: -112.5277605, upper bound: 112.5693728
IS_A1_A2, status: Status.VERIFIED, split count: 2, time: 3.82
Output dim: 3, lower bound: -112.4706063, upper bound: 112.4484261
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.82
Output dim: 3, lower bound: -112.5986886, upper bound: 112.5421197
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 3.82
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -12.3802490, 68.3649139, -14.8509598, 83.8702927, -96.2505417, 83.2158737
1: -20.0955391, 81.2525177, -24.1805515, 99.6739273, -119.7694702, 105.4330673
2: -17.4666786, 81.2615967, -20.9075127, 99.5991058, -117.0657806, 102.1690903
3: -33.8533211, 72.2861710, -40.9821777, 88.5371628, -122.3904877, 113.2683487
4: -25.3386955, 71.4682159, -30.6739483, 87.2144394, -112.5531311, 102.1421509

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4681622, upper bound: 112.4647046
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3745800, upper bound: 112.4132406
time: 0.59 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -14.7116480, 83.1687164, -13.5027428, 74.3206863, -89.0323257, 96.6714630
1: -23.9496403, 98.8321838, -21.9666271, 88.3791046, -112.3287430, 120.7988129
2: -20.7152519, 98.7536392, -19.0338440, 88.3800507, -109.0952911, 117.7874680
3: -40.5850754, 87.7603989, -37.1218758, 78.8891602, -119.4742355, 124.8822708
4: -30.3844452, 86.4514465, -27.7589817, 77.8832703, -108.2677155, 114.2104263

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
time: 0.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.96 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.96
Output dim: 3, lower bound: -112.4681622, upper bound: 112.4647046
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 3.96
Output dim: 3, lower bound: -112.3745800, upper bound: 112.4132406
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.96
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.96
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
Binary search (step 0): status=Status.VERIFIED, low=0.0625000, high=0.1250000, mid=0.0625000, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072838]}

## Binary search (step 1) starts
Candidate diff: 0.0937500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6367131, upper bound: 112.6214954
time: 0.59 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6513812, upper bound: 112.6513812
time: 0.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.50 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 3, lower bound: -112.6367131, upper bound: 112.6214954
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 3, lower bound: -112.6513812, upper bound: 112.6513812

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.6501446, 77.5402985, -14.8509598, 83.8702927, -97.5204315, 92.3912582
1: -22.1772804, 92.1125412, -24.1805515, 99.6739273, -121.8512115, 116.2930908
2: -19.2295589, 92.0258636, -20.9075127, 99.5991058, -118.8286591, 112.9333572
3: -37.4879189, 81.5224380, -40.9821777, 88.5371628, -126.0250549, 122.5046082
4: -28.0775452, 80.3824844, -30.6739483, 87.2144394, -115.2919769, 111.0564270

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5277605, upper bound: 112.5696504
time: 0.54 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4706063, upper bound: 112.4484261
time: 0.60 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -14.7116480, 83.1687164, -14.8509598, 83.8702927, -98.5819321, 98.0196686
1: -23.9496403, 98.8321838, -24.1805515, 99.6739273, -123.6235657, 123.0127258
2: -20.7152519, 98.7536392, -20.9075127, 99.5991058, -120.3143311, 119.6611328
3: -40.5850754, 87.7603989, -40.9821777, 88.5371628, -129.1222229, 128.7425537
4: -30.3844452, 86.4514465, -30.6739483, 87.2144394, -117.5988846, 117.1253967

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
time: 0.59 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
time: 0.70 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.44 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 3, lower bound: -112.5277605, upper bound: 112.5696504
IS_A1_A2, status: Status.VERIFIED, split count: 2, time: 3.44
Output dim: 3, lower bound: -112.4706063, upper bound: 112.4484261
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 3.44
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -12.3802490, 68.3649139, -14.8509598, 83.8702927, -96.2505417, 83.2158737
1: -20.0955391, 81.2525177, -24.1805515, 99.6739273, -119.7694702, 105.4330673
2: -17.4666786, 81.2615967, -20.9075127, 99.5991058, -117.0657806, 102.1690903
3: -33.8533211, 72.2861710, -40.9821777, 88.5371628, -122.3904877, 113.2683487
4: -25.3386955, 71.4682159, -30.6739483, 87.2144394, -112.5531311, 102.1421509

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4881573, upper bound: 112.4751799
time: 0.58 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3745800, upper bound: 112.4132406
time: 0.59 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -14.7116480, 83.1687164, -13.5027428, 74.3206863, -89.0323257, 96.6714630
1: -23.9496403, 98.8321838, -21.9666271, 88.3791046, -112.3287430, 120.7988129
2: -20.7152519, 98.7536392, -19.0338440, 88.3800507, -109.0952911, 117.7874680
3: -40.5850754, 87.7603989, -37.1218758, 78.8891602, -119.4742355, 124.8822708
4: -30.3844452, 86.4514465, -27.7589817, 77.8832703, -108.2677155, 114.2104263

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5936212, upper bound: 112.5387959
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.04 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 4.04
Output dim: 3, lower bound: -112.4881573, upper bound: 112.4751799
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 4.04
Output dim: 3, lower bound: -112.3745800, upper bound: 112.4132406
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.04
Output dim: 3, lower bound: -112.5936212, upper bound: 112.5387959

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -14.7116480, 83.1687164, -11.4839888, 61.1608200, -75.8724518, 94.6527023
1: -23.9496403, 98.8321838, -18.6262779, 72.7687912, -96.7184067, 117.4584656
2: -20.7152519, 98.7536392, -16.2278023, 72.8432999, -93.5585480, 114.9814301
3: -40.5850754, 87.7603989, -31.3023281, 65.3681030, -105.9531708, 119.0627289
4: -30.3844452, 86.4514465, -23.4352016, 64.6841660, -95.0686035, 109.8866501

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -14.7116480, 83.1687164, -13.3482313, 73.5021133, -88.2137451, 96.5169373
1: -23.9496403, 98.8321838, -21.7097340, 87.3989182, -111.3485413, 120.5419159
2: -20.7152519, 98.7536392, -18.8188572, 87.3942261, -108.1094589, 117.5724945
3: -40.5850754, 87.7603989, -36.6799889, 78.0097198, -118.5947876, 124.4403839
4: -30.3844452, 86.4514465, -27.4464264, 77.0085754, -107.3930206, 113.8978729

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5936212, upper bound: 112.5387959
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5936212, upper bound: 112.5387959
time: 0.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.96 seconds
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -112.5936212, upper bound: 112.5387959
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.96
Output dim: 3, lower bound: -112.5936212, upper bound: 112.5387959

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -13.3614035, 73.6087646, -11.4839888, 61.1608200, -74.5222168, 85.0927505
1: -21.7323895, 87.5250168, -18.6262779, 72.7687912, -94.5011826, 106.1512909
2: -18.8386002, 87.5219345, -16.2278023, 72.8432999, -91.6819000, 103.7497253
3: -36.7189484, 78.1003952, -31.3023281, 65.3681030, -102.0870514, 109.4027252
4: -27.4645672, 77.1080704, -23.4352016, 64.6841660, -92.1487350, 100.5432739

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5017162, upper bound: 112.5093595
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5359072, upper bound: 112.4199165
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4041819, upper bound: 112.3729863
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -15.1561556, 85.5653076, -11.4839888, 61.1608200, -76.3169785, 97.0492935
1: -24.7338352, 101.6900330, -18.6262779, 72.7687912, -97.5026169, 120.3163147
2: -21.4997807, 101.5924225, -16.2278023, 72.8432999, -94.3430786, 117.8202133
3: -41.8926926, 90.2411118, -31.3023281, 65.3681030, -107.2607956, 121.5434418
4: -31.4159336, 88.9690552, -23.4352016, 64.6841660, -96.1000900, 112.4042587

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5359072, upper bound: 112.4199165
time: 0.89 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4041819, upper bound: 112.3729863
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -13.3614035, 73.6087646, -13.3482313, 73.5021133, -86.8635101, 86.9569778
1: -21.7323895, 87.5250168, -21.7097340, 87.3989182, -109.1313095, 109.2347488
2: -18.8386002, 87.5219345, -18.8188572, 87.3942261, -106.2328186, 106.3407898
3: -36.7189484, 78.1003952, -36.6799889, 78.0097198, -114.7286682, 114.7803802
4: -27.4645672, 77.1080704, -27.4464264, 77.0085754, -104.4731445, 104.5544968

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5900564, upper bound: 112.5374616
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5337505, upper bound: 112.5088103
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -15.1561556, 85.5653076, -13.3482313, 73.5021133, -88.6582718, 98.9135132
1: -24.7338352, 101.6900330, -21.7097340, 87.3989182, -112.1327438, 123.3997650
2: -21.4997807, 101.5924225, -18.8188572, 87.3942261, -108.8939972, 120.4112778
3: -41.8926926, 90.2411118, -36.6799889, 78.0097198, -119.9024124, 126.9210968
4: -31.4159336, 88.9690552, -27.4464264, 77.0085754, -108.4245071, 116.4154816

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5900564, upper bound: 112.5374616
time: 0.89 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5337505, upper bound: 112.5088103
time: 0.63 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.49 seconds
IS_A2_B1_B1_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.49
Output dim: 3, lower bound: -112.5359072, upper bound: 112.4199165
IS_A2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 5, time: 4.49
Output dim: 3, lower bound: -112.4041819, upper bound: 112.3729863
IS_A2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 4.49
Output dim: 3, lower bound: -112.5359072, upper bound: 112.4199165
IS_A2_B1_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 4.49
Output dim: 3, lower bound: -112.4041819, upper bound: 112.3729863
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.49
Output dim: 3, lower bound: -112.5900564, upper bound: 112.5374616
IS_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 4.49
Output dim: 3, lower bound: -112.5337505, upper bound: 112.5088103
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.49
Output dim: 3, lower bound: -112.5900564, upper bound: 112.5374616
IS_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 4.49
Output dim: 3, lower bound: -112.5337505, upper bound: 112.5088103

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.3614035, 73.6087646, -11.6702394, 61.5867424, -74.9481430, 85.2789993
1: -21.7323895, 87.5250168, -18.9718609, 73.2557983, -94.9881897, 106.4968643
2: -18.8386002, 87.5219345, -16.5078926, 73.3785172, -92.2171097, 104.0298157
3: -36.7189484, 78.1003952, -31.9117374, 65.9289246, -102.6478729, 110.0121307
4: -27.4645672, 77.1080704, -23.8223000, 65.3693848, -92.8339539, 100.9303741

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5526940, upper bound: 112.5526940
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5526940, upper bound: 112.5526940
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -15.1561556, 85.5653076, -11.6702394, 61.5867424, -76.7428970, 97.2355347
1: -24.7338352, 101.6900330, -18.9718609, 73.2557983, -97.9896317, 120.6618881
2: -21.4997807, 101.5924225, -16.5078926, 73.3785172, -94.8782883, 118.1003036
3: -41.8926926, 90.2411118, -31.9117374, 65.9289246, -107.8216171, 122.1528473
4: -31.4159336, 88.9690552, -23.8223000, 65.3693848, -96.7853165, 112.7913513

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5431225, upper bound: 112.4551138
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5267027, upper bound: 112.4123102
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3989935, upper bound: 112.3673562
time: 0.82 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.63 seconds
IS_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 6.63
Output dim: 3, lower bound: -112.5526940, upper bound: 112.5526940
IS_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 6.63
Output dim: 3, lower bound: -112.5526940, upper bound: 112.5526940
IS_A2_B1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 6.63
Output dim: 3, lower bound: -112.5267027, upper bound: 112.4123102
IS_A2_B1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 6.63
Output dim: 3, lower bound: -112.3989935, upper bound: 112.3673562

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.6772413, 61.6686325, -11.6702394, 61.5867424, -73.2639847, 73.3388672
1: -18.9818268, 73.3490524, -18.9718609, 73.2557983, -92.2376251, 92.3209076
2: -16.5200005, 73.4774551, -16.5078926, 73.3785172, -89.8985138, 89.9853439
3: -31.9292965, 65.9926682, -31.9117374, 65.9289246, -97.8582153, 97.9044037
4: -23.8319607, 65.4420929, -23.8223000, 65.3693848, -89.2013321, 89.2643890

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5531052, upper bound: 112.4502965
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4262457, upper bound: 112.4053608
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.6946287, 74.9554520, -11.6702394, 61.5867424, -75.2813721, 86.6256943
1: -22.2909222, 89.1237564, -18.9718609, 73.2557983, -95.5467072, 108.0956039
2: -19.4143085, 89.1291199, -16.5078926, 73.3785172, -92.7928009, 105.6369934
3: -37.6152496, 79.4628983, -31.9117374, 65.9289246, -103.5441589, 111.3746338
4: -28.1466351, 78.6116486, -23.8223000, 65.3693848, -93.5159988, 102.4339447

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5531052, upper bound: 112.4502965
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4262457, upper bound: 112.4053608
time: 0.58 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.18 seconds
IS_A2_B1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 3, lower bound: -112.5531052, upper bound: 112.4502965
IS_A2_B1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 5.18
Output dim: 3, lower bound: -112.4262457, upper bound: 112.4053608
IS_A2_B1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.18
Output dim: 3, lower bound: -112.5531052, upper bound: 112.4502965
IS_A2_B1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.18
Output dim: 3, lower bound: -112.4262457, upper bound: 112.4053608

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.6772413, 61.6686325, -11.2385178, 59.4073677, -71.0846100, 72.9071503
1: -18.9818268, 73.3490524, -18.2584152, 70.6213837, -89.6032104, 91.6074524
2: -16.5200005, 73.4774551, -15.9054604, 70.7503204, -87.2703247, 89.3829117
3: -31.9292965, 65.9926682, -30.6831226, 63.4486809, -95.3779678, 96.6757889
4: -23.8319607, 65.4420929, -22.8884850, 62.9704475, -86.8023834, 88.3305740

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4648531, upper bound: 112.4648531
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4648531, upper bound: 112.4648531
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.6946287, 74.9554520, -11.2385178, 59.4073677, -73.1019897, 86.1939697
1: -22.2909222, 89.1237564, -18.2584152, 70.6213837, -92.9122925, 107.3821487
2: -19.4143085, 89.1291199, -15.9054604, 70.7503204, -90.1646271, 105.0345688
3: -37.6152496, 79.4628983, -30.6831226, 63.4486809, -101.0638962, 110.1460190
4: -28.1466351, 78.6116486, -22.8884850, 62.9704475, -91.1170502, 101.5001373

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5531052, upper bound: 112.4502965
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5440519, upper bound: 112.4460978
time: 0.68 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.80 seconds
IS_A2_B1_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 8, time: 5.80
Output dim: 3, lower bound: -112.4648531, upper bound: 112.4648531
IS_A2_B1_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 8, time: 5.80
Output dim: 3, lower bound: -112.4648531, upper bound: 112.4648531
IS_A2_B1_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.80
Output dim: 3, lower bound: -112.5531052, upper bound: 112.4502965
IS_A2_B1_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.80
Output dim: 3, lower bound: -112.5440519, upper bound: 112.4460978

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -13.6946287, 74.9554520, -10.6966209, 56.6464996, -70.3411255, 85.6520691
1: -22.2909222, 89.1237564, -17.3756695, 67.2999954, -89.5909119, 106.4994278
2: -19.4143085, 89.1291199, -15.1667900, 67.4100113, -86.8243103, 104.2958984
3: -37.6152496, 79.4628983, -29.1840706, 60.4109001, -98.0261307, 108.6469727
4: -28.1466351, 78.6116486, -21.7752571, 59.9726601, -88.1192627, 100.3868942

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5531052, upper bound: 112.4502965
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5394847, upper bound: 112.4272824
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5527851, upper bound: 112.4459048
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -13.6946287, 74.9554520, -11.1743832, 59.0717087, -72.7663269, 86.1298218
1: -22.2909222, 89.1237564, -18.1496830, 70.2188644, -92.5097656, 107.2734375
2: -19.4143085, 89.1291199, -15.8129215, 70.3462296, -89.7605286, 104.9420319
3: -37.6152496, 79.4628983, -30.4986782, 63.0782280, -100.6934586, 109.9615784
4: -28.1466351, 78.6116486, -22.7502995, 62.6060066, -90.7526321, 101.3619461

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5440519, upper bound: 112.4460978
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5183899, upper bound: 112.4333115
time: 0.87 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 6.03 seconds
IS_A2_B1_B2_A1_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 6.03
Output dim: 3, lower bound: -112.5394847, upper bound: 112.4272824
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 6.03
Output dim: 3, lower bound: -112.5527851, upper bound: 112.4459048
IS_A2_B1_B2_A1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 6.03
Output dim: 3, lower bound: -112.5440519, upper bound: 112.4460978
IS_A2_B1_B2_A1_B1_A2_B1_B2_B2, status: Status.VERIFIED, split count: 9, time: 6.03
Output dim: 3, lower bound: -112.5183899, upper bound: 112.4333115

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -13.1849136, 72.0414658, -10.6966209, 56.6464996, -69.8314133, 82.7380829
1: -21.4452095, 85.6512222, -17.3756695, 67.2999954, -88.7452087, 103.0268936
2: -18.7129250, 85.6538544, -15.1667900, 67.4100113, -86.1229401, 100.8206406
3: -36.1691322, 76.4149246, -29.1840706, 60.4109001, -96.5800323, 105.5989914
4: -27.1074123, 75.5891876, -21.7752571, 59.9726601, -87.0800629, 97.3644333

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5394847, upper bound: 112.4272824
time: 0.56 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5102498, upper bound: 112.3517163
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5124194, upper bound: 112.3530620
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -13.4361820, 73.4786606, -10.6966209, 56.6464996, -70.0826797, 84.1752777
1: -21.8560390, 87.3637619, -17.3756695, 67.2999954, -89.1560364, 104.7394333
2: -19.0447731, 87.3766632, -15.1667900, 67.4100113, -86.4547882, 102.5434418
3: -36.8736801, 77.9074020, -29.1840706, 60.4109001, -97.2845764, 107.0914764
4: -27.6081066, 77.0795441, -21.7752571, 59.9726601, -87.5807419, 98.8547974

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5527851, upper bound: 112.4459048
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5386937, upper bound: 112.3995506
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5404336, upper bound: 112.4006040
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -13.6946287, 74.9554520, -9.0324421, 47.0908356, -60.7854614, 83.9878922
1: -22.2909222, 89.1237564, -14.5896549, 55.8870163, -78.1779251, 103.7134018
2: -19.4143085, 89.1291199, -12.8302002, 56.0234489, -75.4377441, 101.9593124
3: -37.6152496, 79.4628983, -24.3779488, 50.2626076, -87.8778534, 103.8408508
4: -28.1466351, 78.6116486, -18.2523689, 50.0445366, -78.1911469, 96.8640137

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5170740, upper bound: 112.4284308
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5432865, upper bound: 112.4417061
time: 0.61 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 4.82 seconds
IS_A2_B1_B2_A1_B1_A2_B1_B1_A1_B1, status: Status.VERIFIED, split count: 10, time: 4.82
Output dim: 3, lower bound: -112.5102498, upper bound: 112.3517163
IS_A2_B1_B2_A1_B1_A2_B1_B1_A1_B2, status: Status.VERIFIED, split count: 10, time: 4.82
Output dim: 3, lower bound: -112.5124194, upper bound: 112.3530620
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 4.82
Output dim: 3, lower bound: -112.5386937, upper bound: 112.3995506
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 4.82
Output dim: 3, lower bound: -112.5404336, upper bound: 112.4006040
IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 10, time: 4.82
Output dim: 3, lower bound: -112.5170740, upper bound: 112.4284308
IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 10, time: 4.82
Output dim: 3, lower bound: -112.5432865, upper bound: 112.4417061

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -13.4361820, 73.4786606, -10.3885202, 55.0713577, -68.5075302, 83.8671799
1: -21.8560390, 87.3637619, -16.8572445, 65.4078598, -87.2639008, 104.2210083
2: -19.0447731, 87.3766632, -14.7274866, 65.5067291, -84.5514984, 102.1041412
3: -36.8736801, 77.9074020, -28.3019600, 58.6760292, -95.5497131, 106.2093506
4: -27.6081066, 77.0795441, -21.1184158, 58.2530861, -85.8611908, 98.1979599

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5404336, upper bound: 112.4006040
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5404336, upper bound: 112.4006040
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5404336, upper bound: 112.4006040
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -13.6946287, 74.9554520, -8.8356991, 46.0971527, -59.7917824, 83.7911530
1: -22.2909222, 89.1237564, -14.2567749, 54.6923294, -76.9832382, 103.3805313
2: -19.4143085, 89.1291199, -12.5467997, 54.8283539, -74.2426605, 101.6759109
3: -37.6152496, 79.4628983, -23.8191166, 49.1714096, -86.7866287, 103.2820053
4: -28.1466351, 78.6116486, -17.8348236, 48.9676857, -77.1142883, 96.4464722

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5432865, upper bound: 112.4412230
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5353589, upper bound: 112.4412062
time: 0.69 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 5.53 seconds
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 11, time: 5.53
Output dim: 3, lower bound: -112.5404336, upper bound: 112.4006040
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 5.53
Output dim: 3, lower bound: -112.5404336, upper bound: 112.4006040
IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 11, time: 5.53
Output dim: 3, lower bound: -112.5432865, upper bound: 112.4412230
IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B2, status: Status.VERIFIED, split count: 11, time: 5.53
Output dim: 3, lower bound: -112.5353589, upper bound: 112.4412062

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.9359608, 70.7755814, -10.3885202, 55.0713577, -68.0073090, 81.1641006
1: -21.0302563, 84.1185226, -16.8572445, 65.4078598, -86.4381104, 100.9757690
2: -18.3502159, 84.1323166, -14.7274866, 65.5067291, -83.8569489, 98.8597946
3: -35.4373894, 74.9316177, -28.3019600, 58.6760292, -94.1134186, 103.2335815
4: -26.5389290, 74.1843414, -21.1184158, 58.2530861, -84.7920151, 95.3027573

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5404336, upper bound: 112.4006040
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5404336, upper bound: 112.4001693
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5334839, upper bound: 112.3991540
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -12.7926664, 68.8334045, -10.3885202, 55.0713577, -67.8640213, 79.2219238
1: -20.8450413, 81.8631821, -16.8572445, 65.4078598, -86.2528992, 98.7204285
2: -18.1101398, 81.9571838, -14.7274866, 65.5067291, -83.6168671, 96.6846619
3: -35.0322151, 73.1023483, -28.3019600, 58.6760292, -93.7082443, 101.4042892
4: -26.1309376, 72.5123596, -21.1184158, 58.2530861, -84.3840103, 93.6307755

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5404336, upper bound: 112.4006040
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5404336, upper bound: 112.4001693
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5334839, upper bound: 112.3991540
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -13.6946287, 74.9554520, -8.3459940, 43.5406799, -57.2353058, 83.3014450
1: -22.2909222, 89.1237564, -13.4515781, 51.6158028, -73.9067230, 102.5753326
2: -19.4143085, 89.1291199, -11.8500004, 51.7415695, -71.1558685, 100.9791107
3: -37.6152496, 79.4628983, -22.4340611, 46.3527527, -83.9679871, 101.8969574
4: -28.1466351, 78.6116486, -16.7852287, 46.1992493, -74.3458786, 95.3968811

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5312032, upper bound: 112.3958854
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5329021, upper bound: 112.3962595
time: 0.98 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 6.23 seconds
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 12, time: 6.23
Output dim: 3, lower bound: -112.5404336, upper bound: 112.4001693
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 12, time: 6.23
Output dim: 3, lower bound: -112.5334839, upper bound: 112.3991540
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 6.23
Output dim: 3, lower bound: -112.5404336, upper bound: 112.4001693
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 12, time: 6.23
Output dim: 3, lower bound: -112.5334839, upper bound: 112.3991540
IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 12, time: 6.23
Output dim: 3, lower bound: -112.5312032, upper bound: 112.3958854
IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B2, status: Status.VERIFIED, split count: 12, time: 6.23
Output dim: 3, lower bound: -112.5329021, upper bound: 112.3962595

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -12.9359608, 70.7755814, -9.8770313, 52.3770828, -65.3130417, 80.6526108
1: -21.0302563, 84.1185226, -16.0146141, 62.1694069, -83.1996536, 100.1331329
2: -18.3502159, 84.1323166, -14.0007076, 62.2539711, -80.6041870, 98.1330261
3: -35.4373894, 74.9316177, -26.8514309, 55.7139091, -91.1512909, 101.7830505
4: -26.5389290, 74.1843414, -20.0234451, 55.3412132, -81.8801422, 94.2077866

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6017989, upper bound: 112.5546547
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5994139, upper bound: 112.5542763
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5994139, upper bound: 112.5546449
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -12.7926664, 68.8334045, -9.8770313, 52.3770828, -65.1697464, 78.7104340
1: -20.8450413, 81.8631821, -16.0146141, 62.1694069, -83.0144501, 97.8777924
2: -18.1101398, 81.9571838, -14.0007076, 62.2539711, -80.3641129, 95.9578934
3: -35.0322151, 73.1023483, -26.8514309, 55.7139091, -90.7461243, 99.9537735
4: -26.1309376, 72.5123596, -20.0234451, 55.3412132, -81.4721527, 92.5358047

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5404336, upper bound: 112.4001693
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5386937, upper bound: 112.4001693
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5386937, upper bound: 112.3991310
time: 0.91 seconds

## Summary of splitting at layer (split count: 12)
- Time for IS candidates: 9.75 seconds
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 13, time: 9.75
Output dim: 3, lower bound: -112.5994139, upper bound: 112.5542763
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 13, time: 9.75
Output dim: 3, lower bound: -112.5994139, upper bound: 112.5546449
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 9.75
Output dim: 3, lower bound: -112.5386937, upper bound: 112.4001693
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 13, time: 9.75
Output dim: 3, lower bound: -112.5386937, upper bound: 112.3991310

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -12.4134645, 67.6239471, -9.8770313, 52.3770828, -64.7905426, 77.5009766
1: -20.1847553, 80.3721695, -16.0146141, 62.1694069, -82.3541641, 96.3867798
2: -17.6119995, 80.4062729, -14.0007076, 62.2539711, -79.8659668, 94.4069824
3: -33.9710732, 71.6419754, -26.8514309, 55.7139091, -89.6849823, 98.4933929
4: -25.4465046, 70.9563751, -20.0234451, 55.3412132, -80.7877045, 90.9798126

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5994139, upper bound: 112.5542763
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5994139, upper bound: 112.5542763
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5994139, upper bound: 112.5542763
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.6136360, 69.0821762, -9.8770313, 52.3770828, -64.9907227, 78.9592056
1: -20.4891071, 82.0864410, -16.0146141, 62.1694069, -82.6585083, 98.1010590
2: -17.8939075, 82.0918350, -14.0007076, 62.2539711, -80.1478729, 96.0925293
3: -34.5136642, 73.0886078, -26.8514309, 55.7139091, -90.2275620, 99.9400253
4: -25.8591881, 72.3533401, -20.0234451, 55.3412132, -81.2003937, 92.3767853

Time for backsubstitution: 2.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5994139, upper bound: 112.5529244
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5994139, upper bound: 112.5546449
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5994139, upper bound: 112.5529244
time: 0.66 seconds

## Summary of splitting at layer (split count: 13)
- Time for IS candidates: 9.40 seconds
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 14, time: 9.40
Output dim: 3, lower bound: -112.5994139, upper bound: 112.5542763
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 14, time: 9.40
Output dim: 3, lower bound: -112.5994139, upper bound: 112.5542763
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 14, time: 9.40
Output dim: 3, lower bound: -112.5994139, upper bound: 112.5546449
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 14, time: 9.40
Output dim: 3, lower bound: -112.5994139, upper bound: 112.5529244

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -11.8270321, 64.3583450, -9.8770313, 52.3770828, -64.2041168, 74.2353745
1: -19.2122059, 76.4685898, -16.0146141, 62.1694069, -81.3816147, 92.4832001
2: -16.7805996, 76.4880676, -14.0007076, 62.2539711, -79.0345688, 90.4887772
3: -32.2855873, 68.1452560, -26.8514309, 55.7139091, -87.9994888, 94.9966888
4: -24.2024498, 67.5036240, -20.0234451, 55.3412132, -79.5436554, 87.5270615

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5542763
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5969312, upper bound: 112.5513488
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 21

Time for candidate selection: 11.45 seconds

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5542763
time: 1.01 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5721650, upper bound: 112.5339169
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5542763
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5542763
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -12.2289658, 66.7053528, -9.8770313, 52.3770828, -64.6060486, 76.5823822
1: -19.8744812, 79.2689819, -16.0146141, 62.1694069, -82.0438843, 95.2835999
2: -17.3426743, 79.3071594, -14.0007076, 62.2539711, -79.5966492, 93.3078690
3: -33.4466896, 70.6301880, -26.8514309, 55.7139091, -89.1605988, 97.4816208
4: -25.0536346, 69.9594803, -20.0234451, 55.3412132, -80.3948364, 89.9829254

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5542763
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5969312, upper bound: 112.5513488
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 21

Time for candidate selection: 11.36 seconds

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5542763
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5721650, upper bound: 112.5339169
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5542763
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5542763
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -12.0119829, 65.7018127, -9.8770313, 52.3770828, -64.3890533, 75.5788422
1: -19.4924545, 78.0465164, -16.0146141, 62.1694069, -81.6618652, 94.0611267
2: -17.0417671, 78.0378952, -14.0007076, 62.2539711, -79.2957230, 92.0386047
3: -32.7837181, 69.4747009, -26.8514309, 55.7139091, -88.4976273, 96.3261261
4: -24.5818596, 68.7893219, -20.0234451, 55.3412132, -79.9230652, 88.8127594

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6005701, upper bound: 112.5546449
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 11

Time for candidate selection: 10.91 seconds

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5756514, upper bound: 112.5279555
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654596, upper bound: 112.5258234
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -12.4266071, 68.1418076, -9.8770313, 52.3770828, -64.8036880, 78.0188370
1: -20.1739960, 80.9575272, -16.0146141, 62.1694069, -82.3433990, 96.9721375
2: -17.6213608, 80.9661255, -14.0007076, 62.2539711, -79.8753281, 94.9668350
3: -33.9805489, 72.0527420, -26.8514309, 55.7139091, -89.6944580, 98.9041748
4: -25.4610825, 71.3337097, -20.0234451, 55.3412132, -80.8022919, 91.3571548

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6005701, upper bound: 112.5546449
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11

Time for candidate selection: 10.92 seconds

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5756514, upper bound: 112.5279554
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654592, upper bound: 112.5258232
time: 0.58 seconds

## Summary of splitting at layer (split count: 14)
- Time for IS candidates: 17.00 seconds
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 15, time: 17.00
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5542763
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 15, time: 17.00
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5542763
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 15, time: 17.00
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5542763
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 15, time: 17.00
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5542763
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 15, time: 17.00
Output dim: 3, lower bound: -112.5756514, upper bound: 112.5279555
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 15, time: 17.00
Output dim: 3, lower bound: -112.5654596, upper bound: 112.5258234
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 15, time: 17.00
Output dim: 3, lower bound: -112.5756514, upper bound: 112.5279554
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 15, time: 17.00
Output dim: 3, lower bound: -112.5654592, upper bound: 112.5258232

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -11.8270321, 64.3583450, -8.2449722, 45.1257553, -56.9527817, 72.6033173
1: -19.2122059, 76.4685898, -13.2232409, 53.4529648, -72.6651688, 89.6918259
2: -16.7805996, 76.4880676, -11.5590076, 53.5650787, -70.3456802, 88.0470657
3: -32.2855873, 68.1452560, -21.9885731, 47.3460350, -79.6316147, 90.1338272
4: -24.2024498, 67.5036240, -16.4293175, 47.0753593, -71.2778091, 83.9329224

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 30

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5772561, upper bound: 112.5437779
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5792205, upper bound: 112.5095774
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6061519, upper bound: 112.5680871
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6039131, upper bound: 112.5671759
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6057550, upper bound: 112.5676439
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -11.8270321, 64.3583450, -9.7369442, 51.6007271, -63.4277534, 74.0952911
1: -19.2122059, 76.4685898, -15.7818909, 61.2419624, -80.4541702, 92.2504654
2: -16.7805996, 76.4880676, -13.7875652, 61.3329620, -78.1135559, 90.2756348
3: -32.2855873, 68.1452560, -26.4501896, 54.8759003, -87.1614685, 94.5954437
4: -24.2024498, 67.5036240, -19.7144527, 54.5208168, -78.7232590, 87.2180634

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6061519, upper bound: 112.5680871
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5772561, upper bound: 112.5439912
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6042432, upper bound: 112.5675161
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6057550, upper bound: 112.5676439
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -12.2289658, 66.7053528, -8.2449722, 45.1257553, -57.3547211, 74.9503250
1: -19.8744812, 79.2689819, -13.2232409, 53.4529648, -73.3274460, 92.4922256
2: -17.3426743, 79.3071594, -11.5590076, 53.5650787, -70.9077530, 90.8661499
3: -33.4466896, 70.6301880, -21.9885731, 47.3460350, -80.7927246, 92.6187592
4: -25.0536346, 69.9594803, -16.4293175, 47.0753593, -72.1289902, 86.3887863

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5721650, upper bound: 112.5337036
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5856710, upper bound: 112.5178177
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5641730, upper bound: 112.4840246
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5542763
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5527502
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5925790, upper bound: 112.5426839
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -12.2289658, 66.7053528, -9.7369442, 51.6007271, -63.8296928, 76.4422989
1: -19.8744812, 79.2689819, -15.7818909, 61.2419624, -81.1164398, 95.0508728
2: -17.3426743, 79.3071594, -13.7875652, 61.3329620, -78.6756287, 93.0947266
3: -33.4466896, 70.6301880, -26.4501896, 54.8759003, -88.3225784, 97.0803757
4: -25.0536346, 69.9594803, -19.7144527, 54.5208168, -79.5744400, 89.6739273

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5641730, upper bound: 112.4840578
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5185836, upper bound: 112.3866561
time: 0.89 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5174369, upper bound: 112.3865359
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -12.0119829, 65.7018127, -8.9590292, 48.1906624, -60.2026443, 74.6608353
1: -19.4924545, 78.0465164, -14.4989233, 57.0854111, -76.5778656, 92.5454330
2: -17.0417671, 78.0378952, -12.7295456, 57.1666412, -74.2084045, 90.7674408
3: -32.7837181, 69.4747009, -24.2175484, 50.7617073, -83.5454254, 93.6922455
4: -24.5818596, 68.7893219, -18.0116692, 50.5281105, -75.1099701, 86.8009720

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 11

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5817883, upper bound: 112.5433919
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5820117, upper bound: 112.5447481
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5820117, upper bound: 112.5447945
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -12.0119829, 65.7018127, -9.7010517, 52.1466522, -64.1586304, 75.4028625
1: -19.4924545, 78.0465164, -15.6143246, 61.8617363, -81.3541870, 93.6608276
2: -17.0417671, 78.0378952, -13.8858299, 61.9852409, -79.0269852, 91.9237213
3: -32.7837181, 69.4747009, -26.3125305, 55.2155342, -87.9992371, 95.7872238
4: -24.5818596, 68.7893219, -19.7153358, 54.7933693, -79.3752136, 88.5046539

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5732929, upper bound: 112.5399322
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5732929, upper bound: 112.5428068
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -12.4266071, 68.1418076, -8.9590292, 48.1906624, -60.6172638, 77.1008377
1: -20.1739960, 80.9575272, -14.4989233, 57.0854111, -77.2593918, 95.4564438
2: -17.6213608, 80.9661255, -12.7295456, 57.1666412, -74.7880020, 93.6956711
3: -33.9805489, 72.0527420, -24.2175484, 50.7617073, -84.7422562, 96.2702942
4: -25.4610825, 71.3337097, -18.0116692, 50.5281105, -75.9891968, 89.3453827

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 44
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5293007, upper bound: 112.4391025
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5756514, upper bound: 112.5279555
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5756514, upper bound: 112.5279090
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5756514, upper bound: 112.5279555
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -12.4266071, 68.1418076, -9.7010517, 52.1466522, -64.5732574, 77.8428574
1: -20.1739960, 80.9575272, -15.6143246, 61.8617363, -82.0357361, 96.5718536
2: -17.6213608, 80.9661255, -13.8858299, 61.9852409, -79.6065826, 94.8519516
3: -33.9805489, 72.0527420, -26.3125305, 55.2155342, -89.1960831, 98.3652649
4: -25.4610825, 71.3337097, -19.7153358, 54.7933693, -80.2544479, 91.0490417

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 44
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5589287, upper bound: 112.5199787
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5589287, upper bound: 112.5258234
time: 0.67 seconds

## Summary of splitting at layer (split count: 15)
- Time for IS candidates: 3.54 seconds
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.6039131, upper bound: 112.5671759
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.6057550, upper bound: 112.5676439
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.6042432, upper bound: 112.5675161
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.6057550, upper bound: 112.5676439
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5527502
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.5925790, upper bound: 112.5426839
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2_B1, status: Status.VERIFIED, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.5185836, upper bound: 112.3866561
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B2_B2, status: Status.VERIFIED, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.5174369, upper bound: 112.3865359
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.5820117, upper bound: 112.5447481
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.5820117, upper bound: 112.5447945
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.5732929, upper bound: 112.5399322
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.5732929, upper bound: 112.5428068
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.5756514, upper bound: 112.5279090
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.5756514, upper bound: 112.5279555
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.5589287, upper bound: 112.5199787
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 16, time: 3.54
Output dim: 3, lower bound: -112.5589287, upper bound: 112.5258234

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -11.8270321, 64.3583450, -7.6438937, 42.0586548, -53.8856812, 72.0022354
1: -19.2122059, 76.4685898, -12.1904106, 49.7459373, -68.9581451, 88.6589966
2: -16.7805996, 76.4880676, -10.6064816, 49.8480873, -66.6286774, 87.0945511
3: -32.2855873, 68.1452560, -20.2385101, 43.9215240, -76.2071075, 88.3837662
4: -24.2024498, 67.5036240, -15.1015720, 43.7116127, -67.9140625, 82.6051941

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6039131, upper bound: 112.5671759
time: 0.97 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6039131, upper bound: 112.5671759
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -11.8270321, 64.3583450, -8.6953869, 46.4684143, -58.2954407, 73.0537262
1: -19.2122059, 76.4685898, -13.9417639, 55.1401176, -74.3523254, 90.4103470
2: -16.7805996, 76.4880676, -12.1301765, 55.2642708, -72.0448608, 88.6182404
3: -32.2855873, 68.1452560, -23.2180500, 49.2740593, -81.5596466, 91.3632965
4: -24.2024498, 67.5036240, -17.4014015, 48.9204025, -73.1228485, 84.9050293

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6057550, upper bound: 112.5676439
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6057550, upper bound: 112.5676439
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -11.8270321, 64.3583450, -9.0896416, 48.2025299, -60.0295563, 73.4479675
1: -19.2122059, 76.4685898, -14.6393213, 57.1544342, -76.3666382, 91.1078873
2: -16.7805996, 76.4880676, -12.7539434, 57.2198868, -74.0004883, 89.2420120
3: -32.2855873, 68.1452560, -24.5045433, 51.1552811, -83.4408493, 92.6497879
4: -24.2024498, 67.5036240, -18.2553711, 50.8419647, -75.0444107, 85.7589874

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 21

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6042432, upper bound: 112.5675161
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6039131, upper bound: 112.5671759
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6042432, upper bound: 112.5675161
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -11.8270321, 64.3583450, -10.2657328, 53.5893517, -65.4163818, 74.6240768
1: -19.2122059, 76.4685898, -16.5964336, 63.6875992, -82.8998032, 93.0650253
2: -16.7805996, 76.4880676, -14.5042229, 63.7607613, -80.5413589, 90.9922714
3: -32.2855873, 68.1452560, -27.8280144, 57.4357834, -89.7213516, 95.9732666
4: -24.2024498, 67.5036240, -20.8368225, 56.9367599, -81.1391983, 88.3404465

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 44
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 21

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6057550, upper bound: 112.5676439
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6057550, upper bound: 112.5676439
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -11.4704685, 62.6312828, -8.2449722, 45.1257553, -56.5962219, 70.8762512
1: -18.5467987, 74.3699493, -13.2232409, 53.4529648, -71.9997559, 87.5931931
2: -16.1525574, 74.3882217, -11.5590076, 53.5650787, -69.7176361, 85.9472122
3: -31.1669121, 66.2115784, -21.9885731, 47.3460350, -78.5129471, 88.2001495
4: -23.3778839, 65.5722580, -16.4293175, 47.0753593, -70.4532394, 82.0015640

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 30

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5706685, upper bound: 112.5334689
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5527502
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6017371, upper bound: 112.5527502
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -12.7598963, 68.5029984, -8.2449722, 45.1257553, -57.8856506, 76.7479630
1: -20.7075253, 81.5079956, -13.2232409, 53.4529648, -74.1604919, 94.7312393
2: -18.0935574, 81.5167160, -11.5590076, 53.5650787, -71.6586380, 93.0757065
3: -34.8767242, 73.0809555, -21.9885731, 47.3460350, -82.2227478, 95.0695267
4: -26.2212086, 72.2570724, -16.4293175, 47.0753593, -73.2965698, 88.6863785

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5641944, upper bound: 112.5247013
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5925790, upper bound: 112.5426839
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5925790, upper bound: 112.5426839
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -12.0119829, 65.7018127, -7.5975804, 41.9623337, -53.9743118, 73.2993927
1: -19.4924545, 78.0465164, -12.1338739, 49.6403580, -69.1328125, 90.1803818
2: -17.0417671, 78.0378952, -10.6686211, 49.7087021, -66.7504730, 88.7065125
3: -32.7837181, 69.4747009, -20.0965405, 43.7061119, -76.4898148, 89.5712433
4: -24.5818596, 68.7893219, -15.0161610, 43.4741592, -68.0560150, 83.8054657

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 44
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 30

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5818127, upper bound: 112.5418734
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5818127, upper bound: 112.5447481
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -12.0119829, 65.7018127, -8.8189430, 47.4279404, -59.4399185, 74.5207443
1: -19.4924545, 78.0465164, -14.2647791, 56.1748848, -75.6673431, 92.3112793
2: -17.0417671, 78.0378952, -12.5192366, 56.2576599, -73.2994232, 90.5571289
3: -32.7837181, 69.4747009, -23.8130989, 49.9356422, -82.7193604, 93.2877960
4: -24.5818596, 68.7893219, -17.7047138, 49.7142296, -74.2960739, 86.4940186

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 11

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5818127, upper bound: 112.5419199
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5818127, upper bound: 112.5447945
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.1333466, 61.5685310, -9.7010517, 52.1466522, -63.2799988, 71.2695770
1: -18.0410328, 73.0504379, -15.6143246, 61.8617363, -79.9027710, 88.6647568
2: -15.8167419, 73.0291672, -13.8858299, 61.9852409, -77.8019714, 86.9149933
3: -30.2462521, 64.6734619, -26.3125305, 55.2155342, -85.4617691, 90.9859924
4: -22.6562595, 64.1043625, -19.7153358, 54.7933693, -77.4496231, 83.8197021

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 44
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5332032, upper bound: 112.4761241
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 19

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5610766, upper bound: 112.5156784
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5614480, upper bound: 112.5179837
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -11.7453947, 65.3140869, -9.7010517, 52.1466522, -63.8920479, 75.0151367
1: -18.9239159, 77.5298996, -15.6143246, 61.8617363, -80.7856522, 93.1442261
2: -16.8047962, 77.5195923, -13.8858299, 61.9852409, -78.7900238, 91.4054184
3: -31.9561901, 68.7231140, -26.3125305, 55.2155342, -87.1717148, 95.0356293
4: -24.1092319, 67.9330063, -19.7153358, 54.7933693, -78.9026031, 87.6483459

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5332032, upper bound: 112.4857275
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 19

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5438660, upper bound: 112.5201866
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5614479, upper bound: 112.5225728
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -12.4266071, 68.1418076, -7.5975804, 41.9623337, -54.3889313, 75.7393875
1: -20.1739960, 80.9575272, -12.1338739, 49.6403580, -69.8143539, 93.0914001
2: -17.6213608, 80.9661255, -10.6686211, 49.7087021, -67.3300629, 91.6347504
3: -33.9805489, 72.0527420, -20.0965405, 43.7061119, -77.6866608, 92.1492844
4: -25.4610825, 71.3337097, -15.0161610, 43.4741592, -68.9352417, 86.3498688

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 44
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5756062, upper bound: 112.5238110
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5756062, upper bound: 112.5279091
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -12.4266071, 68.1418076, -8.8189430, 47.4279404, -59.8545380, 76.9607468
1: -20.1739960, 80.9575272, -14.2647791, 56.1748848, -76.3488693, 95.2222977
2: -17.6213608, 80.9661255, -12.5192366, 56.2576599, -73.8790207, 93.4853592
3: -33.9805489, 72.0527420, -23.8130989, 49.9356422, -83.9161911, 95.8658371
4: -25.4610825, 71.3337097, -17.7047138, 49.7142296, -75.1753006, 89.0384216

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 44
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5293007, upper bound: 112.4391025
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5756062, upper bound: 112.5238574
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5756062, upper bound: 112.5279555
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.5011940, 63.7777367, -9.7010517, 52.1466522, -63.6478386, 73.4787674
1: -18.6477394, 75.6817322, -15.6143246, 61.8617363, -80.5094757, 91.2960587
2: -16.3338432, 75.6802292, -13.8858299, 61.9852409, -78.3190689, 89.5660553
3: -31.3102722, 66.9900818, -26.3125305, 55.2155342, -86.5257950, 93.3026123
4: -23.4377213, 66.3986130, -19.7153358, 54.7933693, -78.2310791, 86.1139450

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 44
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 7

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5332147, upper bound: 112.4761485
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 19

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5343567, upper bound: 112.4985558
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2_A1_B1_A2_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0625000, high=0.0937500, mid=0.0937500, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072838, 112.65165179072838]}

## Binary search (step 2) starts
Candidate diff: 0.0781250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6367131, upper bound: 112.6214954
time: 0.64 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6513812, upper bound: 112.6513812
time: 0.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.62 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 3, lower bound: -112.6367131, upper bound: 112.6214954
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 3, lower bound: -112.6513812, upper bound: 112.6513812

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.6501446, 77.5402985, -14.8509598, 83.8702927, -97.5204315, 92.3912582
1: -22.1772804, 92.1125412, -24.1805515, 99.6739273, -121.8512115, 116.2930908
2: -19.2295589, 92.0258636, -20.9075127, 99.5991058, -118.8286591, 112.9333572
3: -37.4879189, 81.5224380, -40.9821777, 88.5371628, -126.0250549, 122.5046082
4: -28.0775452, 80.3824844, -30.6739483, 87.2144394, -115.2919769, 111.0564270

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5277605, upper bound: 112.5696504
time: 0.58 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4706063, upper bound: 112.4484261
time: 0.59 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -14.7116480, 83.1687164, -14.8509598, 83.8702927, -98.5819321, 98.0196686
1: -23.9496403, 98.8321838, -24.1805515, 99.6739273, -123.6235657, 123.0127258
2: -20.7152519, 98.7536392, -20.9075127, 99.5991058, -120.3143311, 119.6611328
3: -40.5850754, 87.7603989, -40.9821777, 88.5371628, -129.1222229, 128.7425537
4: -30.3844452, 86.4514465, -30.6739483, 87.2144394, -117.5988846, 117.1253967

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
time: 0.61 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
time: 0.82 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.61 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 3, lower bound: -112.5277605, upper bound: 112.5696504
IS_A1_A2, status: Status.VERIFIED, split count: 2, time: 3.61
Output dim: 3, lower bound: -112.4706063, upper bound: 112.4484261
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.61
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 3.61
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -12.3802490, 68.3649139, -14.8509598, 83.8702927, -96.2505417, 83.2158737
1: -20.0955391, 81.2525177, -24.1805515, 99.6739273, -119.7694702, 105.4330673
2: -17.4666786, 81.2615967, -20.9075127, 99.5991058, -117.0657806, 102.1690903
3: -33.8533211, 72.2861710, -40.9821777, 88.5371628, -122.3904877, 113.2683487
4: -25.3386955, 71.4682159, -30.6739483, 87.2144394, -112.5531311, 102.1421509

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4802679, upper bound: 112.4711884
time: 0.56 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3745800, upper bound: 112.4132406
time: 0.61 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -14.7116480, 83.1687164, -13.5027428, 74.3206863, -89.0323257, 96.6714630
1: -23.9496403, 98.8321838, -21.9666271, 88.3791046, -112.3287430, 120.7988129
2: -20.7152519, 98.7536392, -19.0338440, 88.3800507, -109.0952911, 117.7874680
3: -40.5850754, 87.7603989, -37.1218758, 78.8891602, -119.4742355, 124.8822708
4: -30.3844452, 86.4514465, -27.7589817, 77.8832703, -108.2677155, 114.2104263

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5936212, upper bound: 112.5387959
time: 0.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.93 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.93
Output dim: 3, lower bound: -112.4802679, upper bound: 112.4711884
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 3.93
Output dim: 3, lower bound: -112.3745800, upper bound: 112.4132406
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.93
Output dim: 3, lower bound: -112.5936212, upper bound: 112.5387959

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -14.7116480, 83.1687164, -11.4839888, 61.1608200, -75.8724518, 94.6527023
1: -23.9496403, 98.8321838, -18.6262779, 72.7687912, -96.7184067, 117.4584656
2: -20.7152519, 98.7536392, -16.2278023, 72.8432999, -93.5585480, 114.9814301
3: -40.5850754, 87.7603989, -31.3023281, 65.3681030, -105.9531708, 119.0627289
4: -30.3844452, 86.4514465, -23.4352016, 64.6841660, -95.0686035, 109.8866501

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -14.7116480, 83.1687164, -13.3482313, 73.5021133, -88.2137451, 96.5169373
1: -23.9496403, 98.8321838, -21.7097340, 87.3989182, -111.3485413, 120.5419159
2: -20.7152519, 98.7536392, -18.8188572, 87.3942261, -108.1094589, 117.5724945
3: -40.5850754, 87.7603989, -36.6799889, 78.0097198, -118.5947876, 124.4403839
4: -30.3844452, 86.4514465, -27.4464264, 77.0085754, -107.3930206, 113.8978729

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5936212, upper bound: 112.5387959
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5936212, upper bound: 112.5387959
time: 0.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.19 seconds
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -112.5936212, upper bound: 112.5387959
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.19
Output dim: 3, lower bound: -112.5936212, upper bound: 112.5387959

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -13.3614035, 73.6087646, -11.4839888, 61.1608200, -74.5222168, 85.0927505
1: -21.7323895, 87.5250168, -18.6262779, 72.7687912, -94.5011826, 106.1512909
2: -18.8386002, 87.5219345, -16.2278023, 72.8432999, -91.6819000, 103.7497253
3: -36.7189484, 78.1003952, -31.3023281, 65.3681030, -102.0870514, 109.4027252
4: -27.4645672, 77.1080704, -23.4352016, 64.6841660, -92.1487350, 100.5432739

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4983254, upper bound: 112.5018323
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4599623, upper bound: 112.4753281
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4041819, upper bound: 112.3729863
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -15.1561556, 85.5653076, -11.4839888, 61.1608200, -76.3169785, 97.0492935
1: -24.7338352, 101.6900330, -18.6262779, 72.7687912, -97.5026169, 120.3163147
2: -21.4997807, 101.5924225, -16.2278023, 72.8432999, -94.3430786, 117.8202133
3: -41.8926926, 90.2411118, -31.3023281, 65.3681030, -107.2607956, 121.5434418
4: -31.4159336, 88.9690552, -23.4352016, 64.6841660, -96.1000900, 112.4042587

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5346308, upper bound: 112.4199165
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4041819, upper bound: 112.3729863
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -13.3614035, 73.6087646, -13.3482313, 73.5021133, -86.8635101, 86.9569778
1: -21.7323895, 87.5250168, -21.7097340, 87.3989182, -109.1313095, 109.2347488
2: -18.8386002, 87.5219345, -18.8188572, 87.3942261, -106.2328186, 106.3407898
3: -36.7189484, 78.1003952, -36.6799889, 78.0097198, -114.7286682, 114.7803802
4: -27.4645672, 77.1080704, -27.4464264, 77.0085754, -104.4731445, 104.5544968

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5900564, upper bound: 112.5374616
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5337505, upper bound: 112.5088103
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -15.1561556, 85.5653076, -13.3482313, 73.5021133, -88.6582718, 98.9135132
1: -24.7338352, 101.6900330, -21.7097340, 87.3989182, -112.1327438, 123.3997650
2: -21.4997807, 101.5924225, -18.8188572, 87.3942261, -108.8939972, 120.4112778
3: -41.8926926, 90.2411118, -36.6799889, 78.0097198, -119.9024124, 126.9210968
4: -31.4159336, 88.9690552, -27.4464264, 77.0085754, -108.4245071, 116.4154816

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5900564, upper bound: 112.5374616
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5337505, upper bound: 112.5088103
time: 0.65 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 5.06 seconds
IS_A2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 5.06
Output dim: 3, lower bound: -112.4599623, upper bound: 112.4753281
IS_A2_B1_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 5.06
Output dim: 3, lower bound: -112.4041819, upper bound: 112.3729863
IS_A2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 5, time: 5.06
Output dim: 3, lower bound: -112.5346308, upper bound: 112.4199165
IS_A2_B1_B1_A2_B2, status: Status.VERIFIED, split count: 5, time: 5.06
Output dim: 3, lower bound: -112.4041819, upper bound: 112.3729863
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 3, lower bound: -112.5900564, upper bound: 112.5374616
IS_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 5.06
Output dim: 3, lower bound: -112.5337505, upper bound: 112.5088103
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.06
Output dim: 3, lower bound: -112.5900564, upper bound: 112.5374616
IS_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 5.06
Output dim: 3, lower bound: -112.5337505, upper bound: 112.5088103

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -13.3614035, 73.6087646, -11.6702394, 61.5867424, -74.9481430, 85.2789993
1: -21.7323895, 87.5250168, -18.9718609, 73.2557983, -94.9881897, 106.4968643
2: -18.8386002, 87.5219345, -16.5078926, 73.3785172, -92.2171097, 104.0298157
3: -36.7189484, 78.1003952, -31.9117374, 65.9289246, -102.6478729, 110.0121307
4: -27.4645672, 77.1080704, -23.8223000, 65.3693848, -92.8339539, 100.9303741

Time for backsubstitution: 2.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5526940, upper bound: 112.5526940
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5526940, upper bound: 112.5526940
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -15.1561556, 85.5653076, -11.6702394, 61.5867424, -76.7428970, 97.2355347
1: -24.7338352, 101.6900330, -18.9718609, 73.2557983, -97.9896317, 120.6618881
2: -21.4997807, 101.5924225, -16.5078926, 73.3785172, -94.8782883, 118.1003036
3: -41.8926926, 90.2411118, -31.9117374, 65.9289246, -107.8216171, 122.1528473
4: -31.4159336, 88.9690552, -23.8223000, 65.3693848, -96.7853165, 112.7913513

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5431225, upper bound: 112.4551138
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5227676, upper bound: 112.4121059
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3989935, upper bound: 112.3673562
time: 0.60 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 7.01 seconds
IS_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 7.01
Output dim: 3, lower bound: -112.5526940, upper bound: 112.5526940
IS_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 7.01
Output dim: 3, lower bound: -112.5526940, upper bound: 112.5526940
IS_A2_B1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 7.01
Output dim: 3, lower bound: -112.5227676, upper bound: 112.4121059
IS_A2_B1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 6, time: 7.01
Output dim: 3, lower bound: -112.3989935, upper bound: 112.3673562

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.6772413, 61.6686325, -11.6702394, 61.5867424, -73.2639847, 73.3388672
1: -18.9818268, 73.3490524, -18.9718609, 73.2557983, -92.2376251, 92.3209076
2: -16.5200005, 73.4774551, -16.5078926, 73.3785172, -89.8985138, 89.9853439
3: -31.9292965, 65.9926682, -31.9117374, 65.9289246, -97.8582153, 97.9044037
4: -23.8319607, 65.4420929, -23.8223000, 65.3693848, -89.2013321, 89.2643890

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5472119, upper bound: 112.4497531
time: 0.57 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4262457, upper bound: 112.4053608
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -13.6946287, 74.9554520, -11.6702394, 61.5867424, -75.2813721, 86.6256943
1: -22.2909222, 89.1237564, -18.9718609, 73.2557983, -95.5467072, 108.0956039
2: -19.4143085, 89.1291199, -16.5078926, 73.3785172, -92.7928009, 105.6369934
3: -37.6152496, 79.4628983, -31.9117374, 65.9289246, -103.5441589, 111.3746338
4: -28.1466351, 78.6116486, -23.8223000, 65.3693848, -93.5159988, 102.4339447

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5472119, upper bound: 112.4497531
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4262457, upper bound: 112.4053608
time: 0.63 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.22 seconds
IS_A2_B1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.22
Output dim: 3, lower bound: -112.5472119, upper bound: 112.4497531
IS_A2_B1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 3, lower bound: -112.4262457, upper bound: 112.4053608
IS_A2_B1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.22
Output dim: 3, lower bound: -112.5472119, upper bound: 112.4497531
IS_A2_B1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.22
Output dim: 3, lower bound: -112.4262457, upper bound: 112.4053608

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.6772413, 61.6686325, -11.2385178, 59.4073677, -71.0846100, 72.9071503
1: -18.9818268, 73.3490524, -18.2584152, 70.6213837, -89.6032104, 91.6074524
2: -16.5200005, 73.4774551, -15.9054604, 70.7503204, -87.2703247, 89.3829117
3: -31.9292965, 65.9926682, -30.6831226, 63.4486809, -95.3779678, 96.6757889
4: -23.8319607, 65.4420929, -22.8884850, 62.9704475, -86.8023834, 88.3305740

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5624957, upper bound: 112.4979070
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5624957, upper bound: 112.4979070
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -13.6946287, 74.9554520, -11.2385178, 59.4073677, -73.1019897, 86.1939697
1: -22.2909222, 89.1237564, -18.2584152, 70.6213837, -92.9122925, 107.3821487
2: -19.4143085, 89.1291199, -15.9054604, 70.7503204, -90.1646271, 105.0345688
3: -37.6152496, 79.4628983, -30.6831226, 63.4486809, -101.0638962, 110.1460190
4: -28.1466351, 78.6116486, -22.8884850, 62.9704475, -91.1170502, 101.5001373

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5472119, upper bound: 112.4497531
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5401834, upper bound: 112.4460978
time: 0.83 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 6.06 seconds
IS_A2_B1_B2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.06
Output dim: 3, lower bound: -112.5624957, upper bound: 112.4979070
IS_A2_B1_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.06
Output dim: 3, lower bound: -112.5624957, upper bound: 112.4979070
IS_A2_B1_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.06
Output dim: 3, lower bound: -112.5472119, upper bound: 112.4497531
IS_A2_B1_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.06
Output dim: 3, lower bound: -112.5401834, upper bound: 112.4460978

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -11.6772413, 61.6686325, -10.3284960, 54.6452408, -66.3224792, 71.9971313
1: -18.9818268, 73.3490524, -16.7385902, 64.9145508, -83.8963776, 90.0876465
2: -16.5200005, 73.4774551, -14.6334591, 65.0606613, -81.5806503, 88.1109009
3: -31.9292965, 65.9926682, -28.0273266, 58.1003265, -90.0296249, 94.0199966
4: -23.8319607, 65.4420929, -20.9006233, 57.7891502, -81.6211014, 86.3427048

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5624957, upper bound: 112.4979070
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5624957, upper bound: 112.4979070
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -11.6772413, 61.6686325, -11.1218109, 58.8335800, -70.5108185, 72.7904358
1: -18.9818268, 73.3490524, -18.0655937, 69.9301453, -88.9119720, 91.4146423
2: -16.5200005, 73.4774551, -15.7431889, 70.0591888, -86.5791931, 89.2206345
3: -31.9292965, 65.9926682, -30.3531590, 62.8024178, -94.7317123, 96.3458252
4: -23.8319607, 65.4420929, -22.6403103, 62.3399811, -86.1719437, 88.0824051

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5624957, upper bound: 112.4979070
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5624957, upper bound: 112.4979070
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -13.6946287, 74.9554520, -10.6966209, 56.6464996, -70.3411255, 85.6520691
1: -22.2909222, 89.1237564, -17.3756695, 67.2999954, -89.5909119, 106.4994278
2: -19.4143085, 89.1291199, -15.1667900, 67.4100113, -86.8243103, 104.2958984
3: -37.6152496, 79.4628983, -29.1840706, 60.4109001, -98.0261307, 108.6469727
4: -28.1466351, 78.6116486, -21.7752571, 59.9726601, -88.1192627, 100.3868942

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5472119, upper bound: 112.4497531
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5339436, upper bound: 112.4269123
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5467966, upper bound: 112.4453615
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -13.6946287, 74.9554520, -11.1743832, 59.0717087, -72.7663269, 86.1298218
1: -22.2909222, 89.1237564, -18.1496830, 70.2188644, -92.5097656, 107.2734375
2: -19.4143085, 89.1291199, -15.8129215, 70.3462296, -89.7605286, 104.9420319
3: -37.6152496, 79.4628983, -30.4986782, 63.0782280, -100.6934586, 109.9615784
4: -28.1466351, 78.6116486, -22.7502995, 62.6060066, -90.7526321, 101.3619461

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5401834, upper bound: 112.4460978
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5129376, upper bound: 112.4324550
time: 0.67 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 5.83 seconds
IS_A2_B1_B2_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 5.83
Output dim: 3, lower bound: -112.5624957, upper bound: 112.4979070
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.83
Output dim: 3, lower bound: -112.5624957, upper bound: 112.4979070
IS_A2_B1_B2_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 5.83
Output dim: 3, lower bound: -112.5624957, upper bound: 112.4979070
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 5.83
Output dim: 3, lower bound: -112.5624957, upper bound: 112.4979070
IS_A2_B1_B2_A1_B1_A2_B1_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.83
Output dim: 3, lower bound: -112.5339436, upper bound: 112.4269123
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 5.83
Output dim: 3, lower bound: -112.5467966, upper bound: 112.4453615
IS_A2_B1_B2_A1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 9, time: 5.83
Output dim: 3, lower bound: -112.5401834, upper bound: 112.4460978
IS_A2_B1_B2_A1_B1_A2_B1_B2_B2, status: Status.VERIFIED, split count: 9, time: 5.83
Output dim: 3, lower bound: -112.5129376, upper bound: 112.4324550

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -11.2532597, 59.5292015, -10.3284960, 54.6452408, -65.8984909, 69.8576965
1: -18.2805824, 70.7643280, -16.7385902, 64.9145508, -83.1951294, 87.5029144
2: -15.9278469, 70.8979034, -14.6334591, 65.0606613, -80.9885101, 85.5313492
3: -30.7206059, 63.5572472, -28.0273266, 58.1003265, -88.8209305, 91.5845566
4: -22.9130135, 63.0871048, -20.9006233, 57.7891502, -80.7021637, 83.9877319

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5236783, upper bound: 112.4795331
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5078919, upper bound: 112.4558904
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -11.0554247, 57.3105965, -10.3284960, 54.6452408, -65.7006531, 67.6390839
1: -18.0167179, 68.1663284, -16.7385902, 64.9145508, -82.9312668, 84.9049225
2: -15.6248369, 68.3826828, -14.6334591, 65.0606613, -80.6855011, 83.0161362
3: -30.1625900, 61.4011154, -28.0273266, 58.1003265, -88.2629166, 89.4284439
4: -22.4066334, 61.1075096, -20.9006233, 57.7891502, -80.1957855, 82.0081329

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5467094, upper bound: 112.4742642
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5078919, upper bound: 112.4558904
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -11.2532597, 59.5292015, -11.1218109, 58.8335800, -70.0868378, 70.6510086
1: -18.2805824, 70.7643280, -18.0655937, 69.9301453, -88.2107239, 88.8299255
2: -15.9278469, 70.8979034, -15.7431889, 70.0591888, -85.9870377, 86.6410828
3: -30.7206059, 63.5572472, -30.3531590, 62.8024178, -93.5230103, 93.9103851
4: -22.9130135, 63.0871048, -22.6403103, 62.3399811, -85.2529907, 85.7274170

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5375799, upper bound: 112.4805215
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5271465, upper bound: 112.4563020
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -11.0554247, 57.3105965, -11.1218109, 58.8335800, -69.8889923, 68.4323883
1: -18.0167179, 68.1663284, -18.0655937, 69.9301453, -87.9468613, 86.2319183
2: -15.6248369, 68.3826828, -15.7431889, 70.0591888, -85.6840286, 84.1258621
3: -30.1625900, 61.4011154, -30.3531590, 62.8024178, -92.9650116, 91.7542725
4: -22.4066334, 61.1075096, -22.6403103, 62.3399811, -84.7466125, 83.7478180

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5574268, upper bound: 112.4742642
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5271465, upper bound: 112.4563020
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -13.4361820, 73.4786606, -10.6966209, 56.6464996, -70.0826797, 84.1752777
1: -21.8560390, 87.3637619, -17.3756695, 67.2999954, -89.1560364, 104.7394333
2: -19.0447731, 87.3766632, -15.1667900, 67.4100113, -86.4547882, 102.5434418
3: -36.8736801, 77.9074020, -29.1840706, 60.4109001, -97.2845764, 107.0914764
4: -27.6081066, 77.0795441, -21.7752571, 59.9726601, -87.5807419, 98.8547974

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5467966, upper bound: 112.4453615
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5323513, upper bound: 112.3987853
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5342055, upper bound: 112.3998212
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -13.6946287, 74.9554520, -9.0324421, 47.0908356, -60.7854614, 83.9878922
1: -22.2909222, 89.1237564, -14.5896549, 55.8870163, -78.1779251, 103.7134018
2: -19.4143085, 89.1291199, -12.8302002, 56.0234489, -75.4377441, 101.9593124
3: -37.6152496, 79.4628983, -24.3779488, 50.2626076, -87.8778534, 103.8408508
4: -28.1466351, 78.6116486, -18.2523689, 50.0445366, -78.1911469, 96.8640137

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5014249, upper bound: 112.4209286
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5397039, upper bound: 112.4417061
time: 0.69 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 5.10 seconds
IS_A2_B1_B2_A1_B1_A1_B1_B1_A1_A1, status: Status.VERIFIED, split count: 10, time: 5.10
Output dim: 3, lower bound: -112.5236783, upper bound: 112.4795331
IS_A2_B1_B2_A1_B1_A1_B1_B1_A1_A2, status: Status.VERIFIED, split count: 10, time: 5.10
Output dim: 3, lower bound: -112.5078919, upper bound: 112.4558904
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.10
Output dim: 3, lower bound: -112.5467094, upper bound: 112.4742642
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B2, status: Status.VERIFIED, split count: 10, time: 5.10
Output dim: 3, lower bound: -112.5078919, upper bound: 112.4558904
IS_A2_B1_B2_A1_B1_A1_B1_B2_A1_A1, status: Status.VERIFIED, split count: 10, time: 5.10
Output dim: 3, lower bound: -112.5375799, upper bound: 112.4805215
IS_A2_B1_B2_A1_B1_A1_B1_B2_A1_A2, status: Status.VERIFIED, split count: 10, time: 5.10
Output dim: 3, lower bound: -112.5271465, upper bound: 112.4563020
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.10
Output dim: 3, lower bound: -112.5574268, upper bound: 112.4742642
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 5.10
Output dim: 3, lower bound: -112.5271465, upper bound: 112.4563020
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 10, time: 5.10
Output dim: 3, lower bound: -112.5323513, upper bound: 112.3987853
IS_A2_B1_B2_A1_B1_A2_B1_B1_A2_B2, status: Status.VERIFIED, split count: 10, time: 5.10
Output dim: 3, lower bound: -112.5342055, upper bound: 112.3998212
IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 10, time: 5.10
Output dim: 3, lower bound: -112.5014249, upper bound: 112.4209286
IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 10, time: 5.10
Output dim: 3, lower bound: -112.5397039, upper bound: 112.4417061

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -11.0554247, 57.3105965, -8.2466955, 42.9909363, -54.0463600, 65.5572891
1: -18.0167179, 68.1663284, -13.2549314, 50.9771156, -68.9938354, 81.4212418
2: -15.6248369, 68.3826828, -11.7257290, 51.1269493, -66.7517853, 80.1084061
3: -30.1625900, 61.4011154, -22.0498962, 45.6407623, -75.8033524, 83.4510117
4: -22.4066334, 61.1075096, -16.5148106, 45.5474091, -67.9540176, 77.6223221

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4918307, upper bound: 112.4409199
time: 0.59 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5466299, upper bound: 112.4719811
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -11.0554247, 57.3105965, -8.9720097, 46.8102112, -57.8656311, 66.2826080
1: -18.0167179, 68.1663284, -14.4921465, 55.5475731, -73.5642929, 82.6584778
2: -15.6248369, 68.3826828, -12.7482491, 55.6850166, -71.3098526, 81.1309357
3: -30.1625900, 61.4011154, -24.2095280, 49.9389343, -80.1015244, 85.6106415
4: -22.4066334, 61.1075096, -18.1245804, 49.7278595, -72.1344910, 79.2320862

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5113273, upper bound: 112.4451633
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5569664, upper bound: 112.4719811
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -13.6946287, 74.9554520, -8.8356991, 46.0971527, -59.7917824, 83.7911530
1: -22.2909222, 89.1237564, -14.2567749, 54.6923294, -76.9832382, 103.3805313
2: -19.4143085, 89.1291199, -12.5467997, 54.8283539, -74.2426605, 101.6759109
3: -37.6152496, 79.4628983, -23.8191166, 49.1714096, -86.7866287, 103.2820053
4: -28.1466351, 78.6116486, -17.8348236, 48.9676857, -77.1142883, 96.4464722

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5397039, upper bound: 112.4412144
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5353589, upper bound: 112.4412062
time: 0.66 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 5.36 seconds
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 11, time: 5.36
Output dim: 3, lower bound: -112.4918307, upper bound: 112.4409199
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 11, time: 5.36
Output dim: 3, lower bound: -112.5466299, upper bound: 112.4719811
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 11, time: 5.36
Output dim: 3, lower bound: -112.5113273, upper bound: 112.4451633
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 11, time: 5.36
Output dim: 3, lower bound: -112.5569664, upper bound: 112.4719811
IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 11, time: 5.36
Output dim: 3, lower bound: -112.5397039, upper bound: 112.4412144
IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B2, status: Status.VERIFIED, split count: 11, time: 5.36
Output dim: 3, lower bound: -112.5353589, upper bound: 112.4412062

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -11.0554247, 57.3105965, -8.0524035, 42.0248184, -53.0802422, 65.3629990
1: -18.0167179, 68.1663284, -12.9255886, 49.8162193, -67.8329391, 81.0919037
2: -15.6248369, 68.3826828, -11.4459991, 49.9640503, -65.5888901, 79.8286819
3: -30.1625900, 61.4011154, -21.4974823, 44.5781364, -74.7407074, 82.8985977
4: -22.4066334, 61.1075096, -16.1042461, 44.4915428, -66.8981781, 77.2117538

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5466299, upper bound: 112.4719811
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5435562, upper bound: 112.4708679
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -11.0554247, 57.3105965, -8.7800484, 45.8386688, -56.8940887, 66.0906372
1: -18.0167179, 68.1663284, -14.1677895, 54.3797188, -72.3964386, 82.3341064
2: -15.6248369, 68.3826828, -12.4728899, 54.5168915, -70.1417236, 80.8555679
3: -30.1625900, 61.4011154, -23.6653862, 48.8731003, -79.0356903, 85.0664978
4: -22.4066334, 61.1075096, -17.7185650, 48.6760750, -71.0827026, 78.8260727

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5569664, upper bound: 112.4719811
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5515112, upper bound: 112.4708679
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -13.6946287, 74.9554520, -8.3459940, 43.5406799, -57.2353058, 83.3014450
1: -22.2909222, 89.1237564, -13.4515781, 51.6158028, -73.9067230, 102.5753326
2: -19.4143085, 89.1291199, -11.8500004, 51.7415695, -71.1558685, 100.9791107
3: -37.6152496, 79.4628983, -22.4340611, 46.3527527, -83.9679871, 101.8969574
4: -28.1466351, 78.6116486, -16.7852287, 46.1992493, -74.3458786, 95.3968811

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5267107, upper bound: 112.3954899
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5283420, upper bound: 112.3962411
time: 0.87 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 5.65 seconds
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 12, time: 5.65
Output dim: 3, lower bound: -112.5466299, upper bound: 112.4719811
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 12, time: 5.65
Output dim: 3, lower bound: -112.5435562, upper bound: 112.4708679
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 12, time: 5.65
Output dim: 3, lower bound: -112.5569664, upper bound: 112.4719811
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 12, time: 5.65
Output dim: 3, lower bound: -112.5515112, upper bound: 112.4708679
IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 12, time: 5.65
Output dim: 3, lower bound: -112.5267107, upper bound: 112.3954899
IS_A2_B1_B2_A1_B1_A2_B1_B2_B1_B2_B1_B2, status: Status.VERIFIED, split count: 12, time: 5.65
Output dim: 3, lower bound: -112.5283420, upper bound: 112.3962411

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -11.0554247, 57.3105965, -7.4732070, 39.1394043, -50.1948280, 64.7838058
1: -18.0167179, 68.1663284, -11.9841290, 46.3424339, -64.3591537, 80.1504364
2: -15.6248369, 68.3826828, -10.6585436, 46.4686317, -62.0934677, 79.0412292
3: -30.1625900, 61.4011154, -19.8868904, 41.3877754, -71.5503693, 81.2880096
4: -22.4066334, 61.1075096, -14.9089622, 41.3214722, -63.7280998, 76.0164719

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5376780, upper bound: 112.4280641
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5466299, upper bound: 112.4719811
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5466299, upper bound: 112.4719811
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -11.0554247, 57.3105965, -7.9962268, 41.7386513, -52.7940750, 65.3068161
1: -18.0167179, 68.1663284, -12.8300686, 49.4724159, -67.4891357, 80.9963837
2: -15.6248369, 68.3826828, -11.3665257, 49.6189766, -65.2438126, 79.7491989
3: -30.1625900, 61.4011154, -21.3362980, 44.2631989, -74.4257889, 82.7374115
4: -22.4066334, 61.1075096, -15.9849434, 44.1804924, -66.5871048, 77.0924530

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5435562, upper bound: 112.4708679
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5435562, upper bound: 112.4708679
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -11.0554247, 57.3105965, -8.2061014, 43.0074959, -54.0629196, 65.5166779
1: -18.0167179, 68.1663284, -13.2369795, 50.9695206, -68.9862366, 81.4032974
2: -15.6248369, 68.3826828, -11.6930742, 51.0901756, -66.7150116, 80.0757599
3: -30.1625900, 61.4011154, -22.0730305, 45.7392616, -75.9018555, 83.4741440
4: -22.4066334, 61.1075096, -16.5379810, 45.5718422, -67.9784775, 77.6454926

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5569664, upper bound: 112.4719811
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5569664, upper bound: 112.4719811
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -11.0554247, 57.3105965, -8.7188692, 45.5281181, -56.5835381, 66.0294647
1: -18.0167179, 68.1663284, -14.0630932, 54.0059433, -72.0226593, 82.2294083
2: -15.6248369, 68.3826828, -12.3849411, 54.1417389, -69.7665787, 80.7676239
3: -30.1625900, 61.4011154, -23.4885597, 48.5290909, -78.6916809, 84.8896713
4: -22.4066334, 61.1075096, -17.5867329, 48.3369980, -70.7436295, 78.6942444

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5515112, upper bound: 112.4708679
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5515112, upper bound: 112.4708679
time: 0.66 seconds

## Summary of splitting at layer (split count: 12)
- Time for IS candidates: 5.03 seconds
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 13, time: 5.03
Output dim: 3, lower bound: -112.5466299, upper bound: 112.4719811
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 5.03
Output dim: 3, lower bound: -112.5466299, upper bound: 112.4719811
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 13, time: 5.03
Output dim: 3, lower bound: -112.5435562, upper bound: 112.4708679
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 5.03
Output dim: 3, lower bound: -112.5435562, upper bound: 112.4708679
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 13, time: 5.03
Output dim: 3, lower bound: -112.5569664, upper bound: 112.4719811
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 5.03
Output dim: 3, lower bound: -112.5569664, upper bound: 112.4719811
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 13, time: 5.03
Output dim: 3, lower bound: -112.5515112, upper bound: 112.4708679
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 5.03
Output dim: 3, lower bound: -112.5515112, upper bound: 112.4708679

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -8.7227306, 44.4887772, -7.4732070, 39.1394043, -47.8621368, 51.9619827
1: -14.1286507, 52.8057022, -11.9841290, 46.3424339, -60.4710770, 64.7898102
2: -12.3725548, 53.0393944, -10.6585436, 46.4686317, -58.8411865, 63.6979370
3: -23.4894485, 47.5910454, -19.8868904, 41.3877754, -64.8772278, 67.4779205
4: -17.4947281, 47.5769920, -14.9089622, 41.3214722, -58.8162003, 62.4859467

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5376780, upper bound: 112.4280641
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 11

Time for candidate selection: 11.09 seconds

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5466299, upper bound: 112.4719811
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5443100, upper bound: 112.4718829
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -10.9549780, 56.8857956, -7.4732070, 39.1394043, -50.0943832, 64.3590012
1: -17.8377399, 67.6523972, -11.9841290, 46.3424339, -64.1801758, 79.6365128
2: -15.4467907, 67.8677444, -10.6585436, 46.4686317, -61.9154205, 78.5262909
3: -29.8629074, 60.9134979, -19.8868904, 41.3877754, -71.2506866, 80.8003845
4: -22.1705418, 60.6127396, -14.9089622, 41.3214722, -63.4920120, 75.5216980

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5376780, upper bound: 112.4280641
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 22
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 30

Time for candidate selection: 11.79 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5466299, upper bound: 112.4719811
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5443100, upper bound: 112.4718829
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.7227306, 44.4887772, -7.9962268, 41.7386513, -50.4613800, 52.4849968
1: -14.1286507, 52.8057022, -12.8300686, 49.4724159, -63.6010513, 65.6357651
2: -12.3725548, 53.0393944, -11.3665257, 49.6189766, -61.9915314, 64.4059143
3: -23.4894485, 47.5910454, -21.3362980, 44.2631989, -67.7526474, 68.9273376
4: -17.4947281, 47.5769920, -15.9849434, 44.1804924, -61.6752205, 63.5619354

Time for backsubstitution: 2.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5346550, upper bound: 112.4269510
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5357023, upper bound: 112.4661909
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5435562, upper bound: 112.4702466
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -10.9549780, 56.8857956, -7.9962268, 41.7386513, -52.6936302, 64.8820190
1: -17.8377399, 67.6523972, -12.8300686, 49.4724159, -67.3101578, 80.4824677
2: -15.4467907, 67.8677444, -11.3665257, 49.6189766, -65.0657654, 79.2342682
3: -29.8629074, 60.9134979, -21.3362980, 44.2631989, -74.1261063, 82.2497940
4: -22.1705418, 60.6127396, -15.9849434, 44.1804924, -66.3510208, 76.5976868

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5346550, upper bound: 112.4269510
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5357023, upper bound: 112.4661909
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5435562, upper bound: 112.4702466
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -8.7227306, 44.4887772, -8.2061014, 43.0074959, -51.7302246, 52.6948738
1: -14.1286507, 52.8057022, -13.2369795, 50.9695206, -65.0981522, 66.0426788
2: -12.3725548, 53.0393944, -11.6930742, 51.0901756, -63.4627304, 64.7324677
3: -23.4894485, 47.5910454, -22.0730305, 45.7392616, -69.2287140, 69.6640625
4: -17.4947281, 47.5769920, -16.5379810, 45.5718422, -63.0665703, 64.1149673

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5569664, upper bound: 112.4687807
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5555364, upper bound: 112.4713976
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -10.9549780, 56.8857956, -8.2061014, 43.0074959, -53.9624710, 65.0918961
1: -17.8377399, 67.6523972, -13.2369795, 50.9695206, -68.8072586, 80.8893738
2: -15.4467907, 67.8677444, -11.6930742, 51.0901756, -66.5369568, 79.5608215
3: -29.8629074, 60.9134979, -22.0730305, 45.7392616, -75.6021729, 82.9865265
4: -22.1705418, 60.6127396, -16.5379810, 45.5718422, -67.7423859, 77.1507187

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5503836, upper bound: 112.4356643
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5467238, upper bound: 112.4237537
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -8.7227306, 44.4887772, -8.7188692, 45.5281181, -54.2508469, 53.2076340
1: -14.1286507, 52.8057022, -14.0630932, 54.0059433, -68.1345749, 66.8687820
2: -12.3725548, 53.0393944, -12.3849411, 54.1417389, -66.5142975, 65.4243317
3: -23.4894485, 47.5910454, -23.4885597, 48.5290909, -72.0185394, 71.0795822
4: -17.4947281, 47.5769920, -17.5867329, 48.3369980, -65.8317184, 65.1637268

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5515112, upper bound: 112.4674409
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5501014, upper bound: 112.4702466
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -10.9549780, 56.8857956, -8.7188692, 45.5281181, -56.4830933, 65.6046677
1: -17.8377399, 67.6523972, -14.0630932, 54.0059433, -71.8436813, 81.7154922
2: -15.4467907, 67.8677444, -12.3849411, 54.1417389, -69.5885162, 80.2526855
3: -29.8629074, 60.9134979, -23.4885597, 48.5290909, -78.3919983, 84.4020538
4: -22.1705418, 60.6127396, -17.5867329, 48.3369980, -70.5075302, 78.1994705

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5457180, upper bound: 112.4335050
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5422689, upper bound: 112.4216388
time: 0.76 seconds

## Summary of splitting at layer (split count: 13)
- Time for IS candidates: 5.43 seconds
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5466299, upper bound: 112.4719811
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5443100, upper bound: 112.4718829
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5466299, upper bound: 112.4719811
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5443100, upper bound: 112.4718829
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5357023, upper bound: 112.4661909
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5435562, upper bound: 112.4702466
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5357023, upper bound: 112.4661909
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5435562, upper bound: 112.4702466
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5569664, upper bound: 112.4687807
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5555364, upper bound: 112.4713976
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5503836, upper bound: 112.4356643
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5467238, upper bound: 112.4237537
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5515112, upper bound: 112.4674409
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5501014, upper bound: 112.4702466
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5457180, upper bound: 112.4335050
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 14, time: 5.43
Output dim: 3, lower bound: -112.5422689, upper bound: 112.4216388

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.7227306, 44.4887772, -5.0851784, 27.4446125, -36.1673431, 49.5739555
1: -14.1286507, 52.8057022, -8.0223799, 32.3097076, -46.4383545, 60.8280792
2: -12.3725548, 53.0393944, -7.4452114, 32.3906860, -44.7632408, 60.4846039
3: -23.4894485, 47.5910454, -13.1674080, 28.5048599, -51.9943085, 60.7584534
4: -17.4947281, 47.5769920, -10.0521069, 28.4497967, -45.9445190, 57.6290932

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 40
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 19

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 19

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5470201, upper bound: 112.4475778
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5448513, upper bound: 112.4724098
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5624163, upper bound: 112.4979070
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5384309, upper bound: 112.4855557
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5442056, upper bound: 112.4886276
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.7227306, 44.4887772, -7.1100616, 36.9593849, -45.6821136, 51.5988388
1: -14.1286507, 52.8057022, -11.3855267, 43.7376823, -57.8663254, 64.1912079
2: -12.3725548, 53.0393944, -10.1375856, 43.8491592, -56.2217140, 63.1769791
3: -23.4894485, 47.5910454, -18.8616142, 39.0808411, -62.5702896, 66.4526596
4: -17.4947281, 47.5769920, -14.1342392, 39.0574951, -56.5522232, 61.7112274

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 19

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 19

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5170611, upper bound: 112.3906159
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5600963, upper bound: 112.4978087
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5425313, upper bound: 112.4723753
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5447001, upper bound: 112.4474796
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: B, layer: 5, pos: 14
type: A, layer: 5, pos: 44
type: A, layer: 5, pos: 14
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 18
type: B, layer: 5, pos: 18
type: A, layer: 5, pos: 5
type: A, layer: 5, pos: 40
type: B, layer: 5, pos: 5
type: B, layer: 5, pos: 40
type: A, layer: 5, pos: 41
type: B, layer: 5, pos: 41
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 27
type: A, layer: 5, pos: 27
type: A, layer: 5, pos: 46
type: B, layer: 5, pos: 46
type: A, layer: 5, pos: 39
type: B, layer: 5, pos: 15
type: A, layer: 5, pos: 11
type: B, layer: 5, pos: 39
type: B, layer: 5, pos: 11
type: A, layer: 5, pos: 22
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 24
type: B, layer: 5, pos: 13
type: A, layer: 5, pos: 13
type: B, layer: 5, pos: 24
type: A, layer: 5, pos: 2
type: B, layer: 5, pos: 22
type: B, layer: 5, pos: 2
type: A, layer: 5, pos: 0
type: B, layer: 5, pos: 0
type: A, layer: 5, pos: 8
type: B, layer: 5, pos: 8
type: A, layer: 5, pos: 32
type: B, layer: 5, pos: 32
type: A, layer: 5, pos: 38
type: B, layer: 5, pos: 42
type: A, layer: 5, pos: 42
type: B, layer: 5, pos: 38
type: B, layer: 5, pos: 6
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 28
type: A, layer: 5, pos: 30
type: B, layer: 5, pos: 31
type: A, layer: 5, pos: 21
type: A, layer: 5, pos: 31
type: A, layer: 5, pos: 35
type: B, layer: 5, pos: 30
type: B, layer: 5, pos: 21
type: A, layer: 5, pos: 45
type: B, layer: 5, pos: 45
type: B, layer: 5, pos: 35
type: A, layer: 5, pos: 4
type: B, layer: 5, pos: 4
type: B, layer: 5, pos: 29
type: A, layer: 5, pos: 26
type: A, layer: 5, pos: 29
type: A, layer: 5, pos: 34
type: B, layer: 5, pos: 26
type: B, layer: 5, pos: 34
type: A, layer: 5, pos: 36
type: B, layer: 5, pos: 36
type: A, layer: 5, pos: 6
type: A, layer: 5, pos: 10
type: A, layer: 5, pos: 16

Time for candidate selection: 19.33 seconds

### Candidate
type: B, layer: 5, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5529060, upper bound: 112.4866004
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 18

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 18

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 5

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5433454, upper bound: 112.4773823
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5435070, upper bound: 112.4812837
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -10.9549780, 56.8857956, -5.0851784, 27.4446125, -38.3995895, 61.9709740
1: -17.8377399, 67.6523972, -8.0223799, 32.3097076, -50.1474457, 75.6747742
2: -15.4467907, 67.8677444, -7.4452114, 32.3906860, -47.8374786, 75.3129578
3: -29.8629074, 60.9134979, -13.1674080, 28.5048599, -58.3677559, 74.0809021
4: -22.1705418, 60.6127396, -10.0521069, 28.4497967, -50.6203346, 70.6648483

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 12
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 44
type: B, layer: 3, pos: 33
type: A, layer: 3, pos: 40
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 42
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 19

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 19

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5466299, upper bound: 112.4719811
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5459832, upper bound: 112.4622040
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5430409, upper bound: 112.4687558
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5230149, upper bound: 112.4548219
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5287896, upper bound: 112.4578938
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -10.9549780, 56.8857956, -7.1100616, 36.9593849, -47.9143639, 63.9958534
1: -17.8377399, 67.6523972, -11.3855267, 43.7376823, -61.5754242, 79.0379257
2: -15.4467907, 67.8677444, -10.1375856, 43.8491592, -59.2959518, 78.0053329
3: -29.8629074, 60.9134979, -18.8616142, 39.0808411, -68.9437485, 79.7751160
4: -22.1705418, 60.6127396, -14.1342392, 39.0574951, -61.2280350, 74.7469788

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 14
type: A, layer: 3, pos: 14
type: A, layer: 3, pos: 27
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 28
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 12
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 11
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 30

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 28

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 19

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 19

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5110915, upper bound: 112.3811096
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5443100, upper bound: 112.4718829
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5407209, upper bound: 112.4687212
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 42

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 33

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5436633, upper bound: 112.4621057
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 21

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 21

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5433089, upper bound: 112.4654568
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 5
type: A, layer: 5, pos: 44
type: B, layer: 5, pos: 14
type: A, layer: 5, pos: 14
type: B, layer: 5, pos: 44
type: A, layer: 5, pos: 18
type: B, layer: 5, pos: 18
type: A, layer: 5, pos: 40
type: A, layer: 5, pos: 5
type: B, layer: 5, pos: 5
type: B, layer: 5, pos: 40
type: A, layer: 5, pos: 41
type: A, layer: 5, pos: 15
type: B, layer: 5, pos: 41
type: A, layer: 5, pos: 46
type: A, layer: 5, pos: 39
type: B, layer: 5, pos: 46
type: B, layer: 5, pos: 27
type: A, layer: 5, pos: 11
type: A, layer: 5, pos: 27
type: B, layer: 5, pos: 15
type: B, layer: 5, pos: 39
type: A, layer: 5, pos: 22
type: A, layer: 5, pos: 24
type: B, layer: 5, pos: 11
type: A, layer: 5, pos: 25
type: B, layer: 5, pos: 25
type: A, layer: 5, pos: 13
type: B, layer: 5, pos: 13
type: A, layer: 5, pos: 2
type: B, layer: 5, pos: 24
type: B, layer: 5, pos: 2
type: B, layer: 5, pos: 22
type: A, layer: 5, pos: 0
type: B, layer: 5, pos: 0
type: A, layer: 5, pos: 8
type: B, layer: 5, pos: 8
type: A, layer: 5, pos: 32
type: B, layer: 5, pos: 32
type: A, layer: 5, pos: 38
type: B, layer: 5, pos: 6
type: A, layer: 5, pos: 42
type: B, layer: 5, pos: 38
type: B, layer: 5, pos: 42
type: A, layer: 5, pos: 28
type: B, layer: 5, pos: 28
type: A, layer: 5, pos: 30
type: A, layer: 5, pos: 21
type: B, layer: 5, pos: 31
type: A, layer: 5, pos: 35
type: A, layer: 5, pos: 31
type: B, layer: 5, pos: 21
type: B, layer: 5, pos: 30
type: A, layer: 5, pos: 4
type: A, layer: 5, pos: 45
type: B, layer: 5, pos: 45
type: B, layer: 5, pos: 35
type: A, layer: 5, pos: 26
type: B, layer: 5, pos: 4
type: B, layer: 5, pos: 29
type: A, layer: 5, pos: 29
type: A, layer: 5, pos: 34
type: A, layer: 5, pos: 6
type: B, layer: 5, pos: 26
type: A, layer: 5, pos: 36
type: B, layer: 5, pos: 34
type: B, layer: 5, pos: 36
type: A, layer: 5, pos: 10
type: A, layer: 5, pos: 16

Time for candidate selection: 20.71 seconds

### Candidate
type: A, layer: 5, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5413238, upper bound: 112.4607210
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 18

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 5, pos: 18

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 5, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5291125, upper bound: 112.4540538
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5273100, upper bound: 112.4528790
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -8.7227306, 44.4887772, -7.8471994, 41.0523834, -49.7751160, 52.3359756
1: -14.1286507, 52.8057022, -12.5769186, 48.6450424, -62.7736816, 65.3826141
2: -12.3725548, 53.0393944, -11.1466703, 48.7896729, -61.1622276, 64.1860580
3: -23.4894485, 47.5910454, -20.9145622, 43.4873924, -66.9768372, 68.5056000
4: -17.4947281, 47.5769920, -15.6631479, 43.4101524, -60.9048805, 63.2401390

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5529020, upper bound: 112.4575503
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5567492, upper bound: 112.4961725
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5567492, upper bound: 112.4951567
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -10.9549780, 56.8857956, -7.8471994, 41.0523834, -52.0073624, 64.7329941
1: -17.8377399, 67.6523972, -12.5769186, 48.6450424, -66.4827805, 80.2293091
2: -15.4467907, 67.8677444, -11.1466703, 48.7896729, -64.2364655, 79.0144119
3: -29.8629074, 60.9134979, -20.9145622, 43.4873924, -73.3502960, 81.8280640
4: -22.1705418, 60.6127396, -15.6631479, 43.4101524, -65.5806961, 76.2758865

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5346550, upper bound: 112.4263297
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 27
type: B, layer: 3, pos: 22
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 44
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 19
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 19
type: B, layer: 3, pos: 28
type: B, layer: 3, pos: 44
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 33
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 42
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: A, layer: 3, pos: 21
type: A, layer: 3, pos: 46

Time for candidate selection: 11.01 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5435562, upper bound: 112.4702466
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5426483, upper bound: 112.4702032
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -8.7227306, 44.4887772, -7.7055306, 40.3673439, -49.0900726, 52.1943054
1: -14.1286507, 52.8057022, -12.4139090, 47.7892990, -61.9179420, 65.2196121
2: -12.3725548, 53.0393944, -10.9843140, 47.9060516, -60.2786064, 64.0236969
3: -23.4894485, 47.5910454, -20.6592731, 42.8285828, -66.3180313, 68.2502975
4: -17.4947281, 47.5769920, -15.4644108, 42.7202873, -60.2150154, 63.0414047

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5479020, upper bound: 112.4474460
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5386277, upper bound: 112.4213792
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -8.7227306, 44.4887772, -8.0463152, 42.2794571, -51.0021896, 52.5350914
1: -14.1286507, 52.8057022, -12.9687557, 50.0881844, -64.2168121, 65.7744446
2: -12.3725548, 53.0393944, -11.4601812, 50.2134895, -62.5860405, 64.4995728
3: -23.4894485, 47.5910454, -21.6269169, 44.9077721, -68.3972168, 69.2179413
4: -17.4947281, 47.5769920, -16.1936684, 44.7583809, -62.2531090, 63.7706566

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5635840, upper bound: 112.4650273
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5568794, upper bound: 112.4441785
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -10.9549780, 56.8857956, -7.7889481, 40.4940186, -51.4489975, 64.6747437
1: -17.8377399, 67.6523972, -12.5744047, 47.9854431, -65.8231812, 80.2267990
2: -15.4467907, 67.8677444, -11.1109562, 48.1290550, -63.5758438, 78.9786987
3: -29.8629074, 60.9134979, -20.9300213, 43.1311150, -72.9940186, 81.8435211
4: -22.1705418, 60.6127396, -15.6722145, 43.0291595, -65.1996918, 76.2849579

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5458568, upper bound: 112.4330411
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5503835, upper bound: 112.4350702
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -10.9549780, 56.8857956, -7.9148555, 41.4976196, -52.4525986, 64.8006516
1: -17.8377399, 67.6523972, -12.7494373, 49.1549759, -66.9927063, 80.4018326
2: -15.4467907, 67.8677444, -11.2847376, 49.2666168, -64.7134094, 79.1524811
3: -29.8629074, 60.9134979, -21.2423325, 44.0841331, -73.9470367, 82.1558304
4: -22.1705418, 60.6127396, -15.9220905, 43.9339523, -66.1044922, 76.5348282

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5380740, upper bound: 112.4190104
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5467235, upper bound: 112.4231714
time: 0.69 seconds

## Summary of splitting at layer (split count: 14)
- Time for IS candidates: 7.14 seconds
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5384309, upper bound: 112.4855557
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5442056, upper bound: 112.4886276
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5433454, upper bound: 112.4773823
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5435070, upper bound: 112.4812837
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5230149, upper bound: 112.4548219
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5287896, upper bound: 112.4578938
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5291125, upper bound: 112.4540538
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5273100, upper bound: 112.4528790
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5567492, upper bound: 112.4961725
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5567492, upper bound: 112.4951567
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5435562, upper bound: 112.4702466
IS_A2_B1_B2_A1_B1_A1_B1_B1_A2_B1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5426483, upper bound: 112.4702032
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5479020, upper bound: 112.4474460
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B1_B2, status: Status.VERIFIED, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5386277, upper bound: 112.4213792
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5635840, upper bound: 112.4650273
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5568794, upper bound: 112.4441785
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5458568, upper bound: 112.4330411
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5503835, upper bound: 112.4350702
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5380740, upper bound: 112.4190104
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 15, time: 7.14
Output dim: 3, lower bound: -112.5467235, upper bound: 112.4231714
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 14, time: 7.14
Output dim: 3, lower bound: -112.5515112, upper bound: 112.4674409
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 14, time: 7.14
Output dim: 3, lower bound: -112.5501014, upper bound: 112.4702466
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 14, time: 7.14
Output dim: 3, lower bound: -112.5457180, upper bound: 112.4335050
IS_A2_B1_B2_A1_B1_A1_B1_B2_A2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 14, time: 7.14
Output dim: 3, lower bound: -112.5422689, upper bound: 112.4216388
Binary search (step 2): status=Status.UNKNOWN, low=0.0625000, high=0.0781250, mid=0.0781250, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 3) starts
Candidate diff: 0.0703125


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6214954, upper bound: 112.6367131
time: 0.87 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6513812, upper bound: 112.6513812
time: 0.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.67 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 3, lower bound: -112.6214954, upper bound: 112.6367131
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 3, lower bound: -112.6513812, upper bound: 112.6513812

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -14.8509598, 83.8702927, -13.6501446, 77.5402985, -92.3912582, 97.5204315
1: -24.1805515, 99.6739273, -22.1772804, 92.1125412, -116.2930908, 121.8512115
2: -20.9075127, 99.5991058, -19.2295589, 92.0258636, -112.9333649, 118.8286514
3: -40.9821777, 88.5371628, -37.4879189, 81.5224380, -122.5046082, 126.0250549
4: -30.6739483, 87.2144394, -28.0775452, 80.3824844, -111.0564270, 115.2919769

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5696504, upper bound: 112.5277605
time: 0.79 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4484261, upper bound: 112.4706063
time: 0.67 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -14.8509598, 83.8702927, -14.7116480, 83.1687164, -98.0196686, 98.5819321
1: -24.1805515, 99.6739273, -23.9496403, 98.8321838, -123.0127335, 123.6235657
2: -20.9075127, 99.5991058, -20.7152519, 98.7536392, -119.6611328, 120.3143311
3: -40.9821777, 88.5371628, -40.5850754, 87.7603989, -128.7425842, 129.1222229
4: -30.6739483, 87.2144394, -30.3844452, 86.4514465, -117.1253967, 117.5988770

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5426503, upper bound: 112.5995515
time: 0.64 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
time: 0.56 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.35 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 3, lower bound: -112.5696504, upper bound: 112.5277605
IS_B1_B2, status: Status.VERIFIED, split count: 2, time: 3.35
Output dim: 3, lower bound: -112.4484261, upper bound: 112.4706063
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.35
Output dim: 3, lower bound: -112.5426503, upper bound: 112.5995515
IS_B2_A2, status: Status.VERIFIED, split count: 2, time: 3.35
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -14.8509598, 83.8702927, -12.3802490, 68.3649139, -83.2158737, 96.2505417
1: -24.1805515, 99.6739273, -20.0955391, 81.2525177, -105.4330673, 119.7694702
2: -20.9075127, 99.5991058, -17.4666786, 81.2615967, -102.1690903, 117.0657730
3: -40.9821777, 88.5371628, -33.8533211, 72.2861710, -113.2683487, 122.3904877
4: -30.6739483, 87.2144394, -25.3386955, 71.4682159, -102.1421509, 112.5531311

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4680565, upper bound: 112.4756343
time: 0.58 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4132406, upper bound: 112.3745800
time: 0.56 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -13.5027428, 74.3206863, -14.7116480, 83.1687164, -96.6714630, 89.0323257
1: -21.9666271, 88.3791046, -23.9496403, 98.8321838, -120.7988129, 112.3287430
2: -19.0338440, 88.3800507, -20.7152519, 98.7536392, -117.7874680, 109.0952835
3: -37.1218758, 78.8891602, -40.5850754, 87.7603989, -124.8822784, 119.4742355
4: -27.7589817, 77.8832703, -30.3844452, 86.4514465, -114.2104263, 108.2677155

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
time: 0.75 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
time: 0.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.90 seconds
IS_B1_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.90
Output dim: 3, lower bound: -112.4680565, upper bound: 112.4756343
IS_B1_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.90
Output dim: 3, lower bound: -112.4132406, upper bound: 112.3745800
IS_B2_A1_B1, status: Status.VERIFIED, split count: 3, time: 3.90
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
IS_B2_A1_B2, status: Status.VERIFIED, split count: 3, time: 3.90
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
Binary search (step 3): status=Status.VERIFIED, low=0.0703125, high=0.0781250, mid=0.0703125, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 4) starts
Candidate diff: 0.0742188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6367131, upper bound: 112.6214954
time: 0.61 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6513812, upper bound: 112.6513812
time: 0.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.55 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 3, lower bound: -112.6367131, upper bound: 112.6214954
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.55
Output dim: 3, lower bound: -112.6513812, upper bound: 112.6513812

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -13.6501446, 77.5402985, -14.8509598, 83.8702927, -97.5204315, 92.3912582
1: -22.1772804, 92.1125412, -24.1805515, 99.6739273, -121.8512115, 116.2930908
2: -19.2295589, 92.0258636, -20.9075127, 99.5991058, -118.8286591, 112.9333572
3: -37.4879189, 81.5224380, -40.9821777, 88.5371628, -126.0250549, 122.5046082
4: -28.0775452, 80.3824844, -30.6739483, 87.2144394, -115.2919769, 111.0564270

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5277605, upper bound: 112.5696504
time: 0.56 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4706063, upper bound: 112.4484261
time: 0.61 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -14.7116480, 83.1687164, -14.8509598, 83.8702927, -98.5819321, 98.0196686
1: -23.9496403, 98.8321838, -24.1805515, 99.6739273, -123.6235657, 123.0127258
2: -20.7152519, 98.7536392, -20.9075127, 99.5991058, -120.3143311, 119.6611328
3: -40.5850754, 87.7603989, -40.9821777, 88.5371628, -129.1222229, 128.7425537
4: -30.3844452, 86.4514465, -30.6739483, 87.2144394, -117.5988846, 117.1253967

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
time: 0.59 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
time: 0.89 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.65 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -112.5277605, upper bound: 112.5696504
IS_A1_A2, status: Status.VERIFIED, split count: 2, time: 3.65
Output dim: 3, lower bound: -112.4706063, upper bound: 112.4484261
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -112.5995515, upper bound: 112.5426503
IS_A2_B2, status: Status.VERIFIED, split count: 2, time: 3.65
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -12.3802490, 68.3649139, -14.8509598, 83.8702927, -96.2505417, 83.2158737
1: -20.0955391, 81.2525177, -24.1805515, 99.6739273, -119.7694702, 105.4330673
2: -17.4666786, 81.2615967, -20.9075127, 99.5991058, -117.0657806, 102.1690903
3: -33.8533211, 72.2861710, -40.9821777, 88.5371628, -122.3904877, 113.2683487
4: -25.3386955, 71.4682159, -30.6739483, 87.2144394, -112.5531311, 102.1421509

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4780964, upper bound: 112.4697027
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3745800, upper bound: 112.4132406
time: 0.58 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -14.7116480, 83.1687164, -13.5027428, 74.3206863, -89.0323257, 96.6714630
1: -23.9496403, 98.8321838, -21.9666271, 88.3791046, -112.3287430, 120.7988129
2: -20.7152519, 98.7536392, -19.0338440, 88.3800507, -109.0952911, 117.7874680
3: -40.5850754, 87.7603989, -37.1218758, 78.8891602, -119.4742355, 124.8822708
4: -30.3844452, 86.4514465, -27.7589817, 77.8832703, -108.2677155, 114.2104263

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
time: 0.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.01 seconds
IS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 4.01
Output dim: 3, lower bound: -112.4780964, upper bound: 112.4697027
IS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 4.01
Output dim: 3, lower bound: -112.3745800, upper bound: 112.4132406
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.01
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.01
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
Binary search (step 4): status=Status.VERIFIED, low=0.0742188, high=0.0781250, mid=0.0742188, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 5) starts
Candidate diff: 0.0761719


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6214954, upper bound: 112.6367131
time: 0.85 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6513812, upper bound: 112.6513812
time: 0.62 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.66 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.66
Output dim: 3, lower bound: -112.6214954, upper bound: 112.6367131
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.66
Output dim: 3, lower bound: -112.6513812, upper bound: 112.6513812

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -14.8509598, 83.8702927, -13.6501446, 77.5402985, -92.3912582, 97.5204315
1: -24.1805515, 99.6739273, -22.1772804, 92.1125412, -116.2930908, 121.8512115
2: -20.9075127, 99.5991058, -19.2295589, 92.0258636, -112.9333649, 118.8286514
3: -40.9821777, 88.5371628, -37.4879189, 81.5224380, -122.5046082, 126.0250549
4: -30.6739483, 87.2144394, -28.0775452, 80.3824844, -111.0564270, 115.2919769

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_B1_B1

### Relational analysis result of IS_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5696504, upper bound: 112.5277605
time: 0.77 seconds

## Relational analysis of IS_B1_B2

### Relational analysis result of IS_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4484261, upper bound: 112.4706063
time: 0.75 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -14.8509598, 83.8702927, -14.7116480, 83.1687164, -98.0196686, 98.5819321
1: -24.1805515, 99.6739273, -23.9496403, 98.8321838, -123.0127335, 123.6235657
2: -20.9075127, 99.5991058, -20.7152519, 98.7536392, -119.6611328, 120.3143311
3: -40.9821777, 88.5371628, -40.5850754, 87.7603989, -128.7425842, 129.1222229
4: -30.6739483, 87.2144394, -30.3844452, 86.4514465, -117.1253967, 117.5988770

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5426503, upper bound: 112.5995515
time: 0.62 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030
time: 0.58 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.38 seconds
IS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 3, lower bound: -112.5696504, upper bound: 112.5277605
IS_B1_B2, status: Status.VERIFIED, split count: 2, time: 3.38
Output dim: 3, lower bound: -112.4484261, upper bound: 112.4706063
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.38
Output dim: 3, lower bound: -112.5426503, upper bound: 112.5995515
IS_B2_A2, status: Status.VERIFIED, split count: 2, time: 3.38
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4909030

## BFS IS instance: IS_B1_B1

### Backsubstitution after applying IS history:
0: -14.8509598, 83.8702927, -12.3802490, 68.3649139, -83.2158737, 96.2505417
1: -24.1805515, 99.6739273, -20.0955391, 81.2525177, -105.4330673, 119.7694702
2: -20.9075127, 99.5991058, -17.4666786, 81.2615967, -102.1690903, 117.0657730
3: -40.9821777, 88.5371628, -33.8533211, 72.2861710, -113.2683487, 122.3904877
4: -30.6739483, 87.2144394, -25.3386955, 71.4682159, -102.1421509, 112.5531311

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B1_B1_A1

### Relational analysis result of IS_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4704562, upper bound: 112.4791949
time: 0.60 seconds

## Relational analysis of IS_B1_B1_A2

### Relational analysis result of IS_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4132406, upper bound: 112.3745800
time: 0.59 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -13.5027428, 74.3206863, -14.7116480, 83.1687164, -96.6714630, 89.0323257
1: -21.9666271, 88.3791046, -23.9496403, 98.8321838, -120.7988129, 112.3287430
2: -19.0338440, 88.3800507, -20.7152519, 98.7536392, -117.7874680, 109.0952835
3: -37.1218758, 78.8891602, -40.5850754, 87.7603989, -124.8822784, 119.4742355
4: -27.7589817, 77.8832703, -30.3844452, 86.4514465, -114.2104263, 108.2677155

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5426503, upper bound: 112.5995515
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5387959, upper bound: 112.5936212
time: 0.85 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.28 seconds
IS_B1_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.28
Output dim: 3, lower bound: -112.4704562, upper bound: 112.4791949
IS_B1_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.28
Output dim: 3, lower bound: -112.4132406, upper bound: 112.3745800
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 3, lower bound: -112.5426503, upper bound: 112.5995515
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.28
Output dim: 3, lower bound: -112.5387959, upper bound: 112.5936212

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -11.4839888, 61.1608200, -14.7116480, 83.1687164, -94.6527023, 75.8724518
1: -18.6262779, 72.7687912, -23.9496403, 98.8321838, -117.4584656, 96.7184067
2: -16.2278023, 72.8432999, -20.7152519, 98.7536392, -114.9814301, 93.5585480
3: -31.3023281, 65.3681030, -40.5850754, 87.7603989, -119.0627289, 105.9531708
4: -23.4352016, 64.6841660, -30.3844452, 86.4514465, -109.8866501, 95.0686035

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5426503, upper bound: 112.5995515
time: 0.72 seconds

## Relational analysis of IS_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5426503, upper bound: 112.5995515
time: 0.84 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -13.3482313, 73.5021133, -14.7116480, 83.1687164, -96.5169449, 88.2137451
1: -21.7097340, 87.3989182, -23.9496403, 98.8321838, -120.5419159, 111.3485413
2: -18.8188572, 87.3942261, -20.7152519, 98.7536392, -117.5724945, 108.1094589
3: -36.6799889, 78.0097198, -40.5850754, 87.7603989, -124.4403839, 118.5947952
4: -27.4464264, 77.0085754, -30.3844452, 86.4514465, -113.8978729, 107.3930206

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_B2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5387959, upper bound: 112.5936212
time: 0.84 seconds

## Relational analysis of IS_B2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5387959, upper bound: 112.5936212
time: 0.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.12 seconds
IS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -112.5426503, upper bound: 112.5995515
IS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -112.5426503, upper bound: 112.5995515
IS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -112.5387959, upper bound: 112.5936212
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.12
Output dim: 3, lower bound: -112.5387959, upper bound: 112.5936212

## BFS IS instance: IS_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: -11.4839888, 61.1608200, -13.3614035, 73.6087646, -85.0927505, 74.5222092
1: -18.6262779, 72.7687912, -21.7323895, 87.5250168, -106.1512909, 94.5011826
2: -16.2278023, 72.8432999, -18.8386002, 87.5219345, -103.7497253, 91.6819000
3: -31.3023281, 65.3681030, -36.7189484, 78.1003952, -109.4027252, 102.0870514
4: -23.4352016, 64.6841660, -27.4645672, 77.1080704, -100.5432739, 92.1487350

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A1_B1_B1

### Relational analysis result of IS_B2_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5007945, upper bound: 112.4975976
time: 0.58 seconds

## Relational analysis of IS_B2_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_A1_B1_A1

### Relational analysis result of IS_B2_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4199165, upper bound: 112.5343782
time: 0.62 seconds

## Relational analysis of IS_B2_A1_A1_B1_A2

### Relational analysis result of IS_B2_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3729863, upper bound: 112.4041819
time: 0.59 seconds

## BFS IS instance: IS_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -11.4839888, 61.1608200, -15.1561556, 85.5653076, -97.0492935, 76.3169785
1: -18.6262779, 72.7687912, -24.7338352, 101.6900330, -120.3163147, 97.5026169
2: -16.2278023, 72.8432999, -21.4997807, 101.5924225, -117.8202057, 94.3430786
3: -31.3023281, 65.3681030, -41.8926926, 90.2411118, -121.5434418, 107.2607956
4: -23.4352016, 64.6841660, -31.4159336, 88.9690552, -112.4042587, 96.1000900

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4199165, upper bound: 112.5343782
time: 0.64 seconds

## Relational analysis of IS_B2_A1_A1_B2_A2

### Relational analysis result of IS_B2_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3729863, upper bound: 112.4041819
time: 0.63 seconds

## BFS IS instance: IS_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -13.3482313, 73.5021133, -13.3614035, 73.6087646, -86.9569778, 86.8635101
1: -21.7097340, 87.3989182, -21.7323895, 87.5250168, -109.2347488, 109.1313095
2: -18.8188572, 87.3942261, -18.8386002, 87.5219345, -106.3407898, 106.2328186
3: -36.6799889, 78.0097198, -36.7189484, 78.1003952, -114.7803802, 114.7286682
4: -27.4464264, 77.0085754, -27.4645672, 77.1080704, -104.5544968, 104.4731445

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_A2_B1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5374616, upper bound: 112.5900564
time: 0.67 seconds

## Relational analysis of IS_B2_A1_A2_B1_A2

### Relational analysis result of IS_B2_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5088103, upper bound: 112.5337505
time: 0.78 seconds

## BFS IS instance: IS_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -13.3482313, 73.5021133, -15.1561556, 85.5653076, -98.9135132, 88.6582718
1: -21.7097340, 87.3989182, -24.7338352, 101.6900330, -123.3997650, 112.1327515
2: -18.8188572, 87.3942261, -21.4997807, 101.5924225, -120.4112701, 108.8939972
3: -36.6799889, 78.0097198, -41.8926926, 90.2411118, -126.9210968, 119.9024124
4: -27.4464264, 77.0085754, -31.4159336, 88.9690552, -116.4154816, 108.4245071

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_A2_B2_A1

### Relational analysis result of IS_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5374616, upper bound: 112.5900564
time: 0.62 seconds

## Relational analysis of IS_B2_A1_A2_B2_A2

### Relational analysis result of IS_B2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5088103, upper bound: 112.5337505
time: 0.77 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.41 seconds
IS_B2_A1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 3, lower bound: -112.4199165, upper bound: 112.5343782
IS_B2_A1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 3, lower bound: -112.3729863, upper bound: 112.4041819
IS_B2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 3, lower bound: -112.4199165, upper bound: 112.5343782
IS_B2_A1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 3, lower bound: -112.3729863, upper bound: 112.4041819
IS_B2_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -112.5374616, upper bound: 112.5900564
IS_B2_A1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 3, lower bound: -112.5088103, upper bound: 112.5337505
IS_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.41
Output dim: 3, lower bound: -112.5374616, upper bound: 112.5900564
IS_B2_A1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.41
Output dim: 3, lower bound: -112.5088103, upper bound: 112.5337505

## BFS IS instance: IS_B2_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -11.6702394, 61.5867424, -13.3614035, 73.6087646, -85.2789993, 74.9481430
1: -18.9718609, 73.2557983, -21.7323895, 87.5250168, -106.4968643, 94.9881897
2: -16.5078926, 73.3785172, -18.8386002, 87.5219345, -104.0298157, 92.2171097
3: -31.9117374, 65.9289246, -36.7189484, 78.1003952, -110.0121307, 102.6478729
4: -23.8223000, 65.3693848, -27.4645672, 77.1080704, -100.9303741, 92.8339539

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_A2_B1_A1_B1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5526940, upper bound: 112.5526940
time: 0.66 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5526940, upper bound: 112.5526940
time: 0.68 seconds

## BFS IS instance: IS_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -11.6702394, 61.5867424, -15.1561556, 85.5653076, -97.2355347, 76.7428970
1: -18.9718609, 73.2557983, -24.7338352, 101.6900330, -120.6618805, 97.9896317
2: -16.5078926, 73.3785172, -21.4997807, 101.5924225, -118.1003113, 94.8782883
3: -31.9117374, 65.9289246, -41.8926926, 90.2411118, -122.1528473, 107.8216171
4: -23.8223000, 65.3693848, -31.4159336, 88.9690552, -112.7913513, 96.7853165

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B2_A1_A1

### Relational analysis result of IS_B2_A1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4551138, upper bound: 112.5431225
time: 0.66 seconds

## Relational analysis of IS_B2_A1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_B2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_A2_B2_A1_A1

### Relational analysis result of IS_B2_A1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4120081, upper bound: 112.5219990
time: 0.90 seconds

## Relational analysis of IS_B2_A1_A2_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3673562, upper bound: 112.3989935
time: 0.61 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 6.55 seconds
IS_B2_A1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.55
Output dim: 3, lower bound: -112.5526940, upper bound: 112.5526940
IS_B2_A1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.55
Output dim: 3, lower bound: -112.5526940, upper bound: 112.5526940
IS_B2_A1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 6.55
Output dim: 3, lower bound: -112.4120081, upper bound: 112.5219990
IS_B2_A1_A2_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 6.55
Output dim: 3, lower bound: -112.3673562, upper bound: 112.3989935

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -11.6702394, 61.5867424, -11.6772413, 61.6686325, -73.3388748, 73.2639847
1: -18.9718609, 73.2557983, -18.9818268, 73.3490524, -92.3209076, 92.2376251
2: -16.5078926, 73.3785172, -16.5200005, 73.4774551, -89.9853439, 89.8985138
3: -31.9117374, 65.9289246, -31.9292965, 65.9926682, -97.9044037, 97.8582153
4: -23.8223000, 65.3693848, -23.8319607, 65.4420929, -89.2643890, 89.2013321

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4496553, upper bound: 112.5464426
time: 0.64 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4053608, upper bound: 112.4262457
time: 0.98 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -11.6702394, 61.5867424, -13.6946287, 74.9554520, -86.6256943, 75.2813721
1: -18.9718609, 73.2557983, -22.2909222, 89.1237564, -108.0956039, 95.5467072
2: -16.5078926, 73.3785172, -19.4143085, 89.1291199, -105.6369934, 92.7928009
3: -31.9117374, 65.9289246, -37.6152496, 79.4628983, -111.3746338, 103.5441589
4: -23.8223000, 65.3693848, -28.1466351, 78.6116486, -102.4339447, 93.5159988

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4496553, upper bound: 112.5464426
time: 0.73 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4053608, upper bound: 112.4262457
time: 0.82 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.62 seconds
IS_B2_A1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.62
Output dim: 3, lower bound: -112.4496553, upper bound: 112.5464426
IS_B2_A1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.62
Output dim: 3, lower bound: -112.4053608, upper bound: 112.4262457
IS_B2_A1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.62
Output dim: 3, lower bound: -112.4496553, upper bound: 112.5464426
IS_B2_A1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.62
Output dim: 3, lower bound: -112.4053608, upper bound: 112.4262457

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -11.2385178, 59.4073677, -11.6772413, 61.6686325, -72.9071503, 71.0846100
1: -18.2584152, 70.6213837, -18.9818268, 73.3490524, -91.6074524, 89.6032104
2: -15.9054604, 70.7503204, -16.5200005, 73.4774551, -89.3829117, 87.2703247
3: -30.6831226, 63.4486809, -31.9292965, 65.9926682, -96.6757889, 95.3779602
4: -22.8884850, 62.9704475, -23.8319607, 65.4420929, -88.3305817, 86.8023834

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4979070, upper bound: 112.5624957
time: 0.96 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4979070, upper bound: 112.5768128
time: 0.62 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -11.2385178, 59.4073677, -13.6946287, 74.9554520, -86.1939697, 73.1019897
1: -18.2584152, 70.6213837, -22.2909222, 89.1237564, -107.3821564, 92.9122925
2: -15.9054604, 70.7503204, -19.4143085, 89.1291199, -105.0345688, 90.1646271
3: -30.6831226, 63.4486809, -37.6152496, 79.4628983, -110.1460190, 101.0638962
4: -22.8884850, 62.9704475, -28.1466351, 78.6116486, -101.5001373, 91.1170502

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4496553, upper bound: 112.5464426
time: 0.64 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4460978, upper bound: 112.5396031
time: 0.78 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 6.38 seconds
IS_B2_A1_A2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 6.38
Output dim: 3, lower bound: -112.4979070, upper bound: 112.5624957
IS_B2_A1_A2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 6.38
Output dim: 3, lower bound: -112.4979070, upper bound: 112.5768128
IS_B2_A1_A2_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 6.38
Output dim: 3, lower bound: -112.4496553, upper bound: 112.5464426
IS_B2_A1_A2_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 6.38
Output dim: 3, lower bound: -112.4460978, upper bound: 112.5396031

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -10.3284960, 54.6452408, -11.6772413, 61.6686325, -71.9971237, 66.3224792
1: -16.7385902, 64.9145508, -18.9818268, 73.3490524, -90.0876465, 83.8963776
2: -14.6334591, 65.0606613, -16.5200005, 73.4774551, -88.1109009, 81.5806503
3: -28.0273266, 58.1003265, -31.9292965, 65.9926682, -94.0199966, 90.0296249
4: -20.9006233, 57.7891502, -23.8319607, 65.4420929, -86.3427048, 81.6211014

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4979070, upper bound: 112.5624957
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4979070, upper bound: 112.5624957
time: 0.92 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -11.1218109, 58.8335800, -11.6772413, 61.6686325, -72.7904358, 70.5108185
1: -18.0655937, 69.9301453, -18.9818268, 73.3490524, -91.4146423, 88.9119720
2: -15.7431889, 70.0591888, -16.5200005, 73.4774551, -89.2206345, 86.5791931
3: -30.3531590, 62.8024178, -31.9292965, 65.9926682, -96.3458252, 94.7317123
4: -22.6403103, 62.3399811, -23.8319607, 65.4420929, -88.0824051, 86.1719437

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4979070, upper bound: 112.5768128
time: 0.57 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4979070, upper bound: 112.5768128
time: 0.61 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -10.6966209, 56.6464996, -13.6946287, 74.9554520, -85.6520691, 70.3411255
1: -17.3756695, 67.2999954, -22.2909222, 89.1237564, -106.4994202, 89.5909195
2: -15.1667900, 67.4100113, -19.4143085, 89.1291199, -104.2958984, 86.8243103
3: -29.1840706, 60.4109001, -37.6152496, 79.4628983, -108.6469727, 98.0261383
4: -21.7752571, 59.9726601, -28.1466351, 78.6116486, -100.3868942, 88.1192703

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4496553, upper bound: 112.5464426
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4268145, upper bound: 112.5331775
time: 0.87 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4452636, upper bound: 112.5460305
time: 0.64 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -11.1743832, 59.0717087, -13.6946287, 74.9554520, -86.1298218, 72.7663269
1: -18.1496830, 70.2188644, -22.2909222, 89.1237564, -107.2734375, 92.5097656
2: -15.8129215, 70.3462296, -19.4143085, 89.1291199, -104.9420319, 89.7605286
3: -30.4986782, 63.0782280, -37.6152496, 79.4628983, -109.9615784, 100.6934662
4: -22.7502995, 62.6060066, -28.1466351, 78.6116486, -101.3619461, 90.7526321

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4460978, upper bound: 112.5396031
time: 0.66 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4323240, upper bound: 112.5121185
time: 0.63 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 5.78 seconds
IS_B2_A1_A2_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 9, time: 5.78
Output dim: 3, lower bound: -112.4979070, upper bound: 112.5624957
IS_B2_A1_A2_B1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 5.78
Output dim: 3, lower bound: -112.4979070, upper bound: 112.5624957
IS_B2_A1_A2_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 5.78
Output dim: 3, lower bound: -112.4979070, upper bound: 112.5768128
IS_B2_A1_A2_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 5.78
Output dim: 3, lower bound: -112.4979070, upper bound: 112.5768128
IS_B2_A1_A2_B1_A1_B2_A1_A1_B1, status: Status.VERIFIED, split count: 9, time: 5.78
Output dim: 3, lower bound: -112.4268145, upper bound: 112.5331775
IS_B2_A1_A2_B1_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 9, time: 5.78
Output dim: 3, lower bound: -112.4452636, upper bound: 112.5460305
IS_B2_A1_A2_B1_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 5.78
Output dim: 3, lower bound: -112.4460978, upper bound: 112.5396031
IS_B2_A1_A2_B1_A1_B2_A1_A2_A2, status: Status.VERIFIED, split count: 9, time: 5.78
Output dim: 3, lower bound: -112.4323240, upper bound: 112.5121185

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -10.3284960, 54.6452408, -11.2532597, 59.5292015, -69.8576965, 65.8984985
1: -16.7385902, 64.9145508, -18.2805824, 70.7643280, -87.5029144, 83.1951294
2: -14.6334591, 65.0606613, -15.9278469, 70.8979034, -85.5313492, 80.9885101
3: -28.0273266, 58.1003265, -30.7206059, 63.5572472, -91.5845566, 88.8209229
4: -20.9006233, 57.7891502, -22.9130135, 63.0871048, -83.9877319, 80.7021637

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B1_B1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4795331, upper bound: 112.5236783
time: 0.63 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B1_B2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4558904, upper bound: 112.5078919
time: 1.27 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -10.3284960, 54.6452408, -11.0554247, 57.3105965, -67.6390762, 65.7006454
1: -16.7385902, 64.9145508, -18.0167179, 68.1663284, -84.9049225, 82.9312668
2: -14.6334591, 65.0606613, -15.6248369, 68.3826828, -83.0161362, 80.6855011
3: -28.0273266, 58.1003265, -30.1625900, 61.4011154, -89.4284439, 88.2629166
4: -20.9006233, 57.7891502, -22.4066334, 61.1075096, -82.0081329, 80.1957855

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4742642, upper bound: 112.5467094
time: 0.73 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4558904, upper bound: 112.5078919
time: 0.68 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -11.1218109, 58.8335800, -11.2532597, 59.5292015, -70.6510010, 70.0868378
1: -18.0655937, 69.9301453, -18.2805824, 70.7643280, -88.8299255, 88.2107239
2: -15.7431889, 70.0591888, -15.9278469, 70.8979034, -86.6410828, 85.9870377
3: -30.3531590, 62.8024178, -30.7206059, 63.5572472, -93.9103928, 93.5230103
4: -22.6403103, 62.3399811, -22.9130135, 63.0871048, -85.7274170, 85.2529907

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4885314, upper bound: 112.5402522
time: 0.60 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4674642, upper bound: 112.5316413
time: 0.95 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -11.1218109, 58.8335800, -11.0554247, 57.3105965, -68.4323883, 69.8889923
1: -18.0655937, 69.9301453, -18.0167179, 68.1663284, -86.2319107, 87.9468613
2: -15.7431889, 70.0591888, -15.6248369, 68.3826828, -84.1258621, 85.6840286
3: -30.3531590, 62.8024178, -30.1625900, 61.4011154, -91.7542725, 92.9650116
4: -22.6403103, 62.3399811, -22.4066334, 61.1075096, -83.7478180, 84.7466125

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4851516, upper bound: 112.5664392
time: 0.65 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4674642, upper bound: 112.5316413
time: 0.84 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: -10.6966209, 56.6464996, -13.4361820, 73.4786606, -84.1752777, 70.0826797
1: -17.3756695, 67.2999954, -21.8560390, 87.3637619, -104.7394333, 89.1560364
2: -15.1667900, 67.4100113, -19.0447731, 87.3766632, -102.5434418, 86.4547882
3: -29.1840706, 60.4109001, -36.8736801, 77.9074020, -107.0914764, 97.2845764
4: -21.7752571, 59.9726601, -27.6081066, 77.0795441, -98.8547974, 87.5807495

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4452636, upper bound: 112.5460305
time: 0.63 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_B2_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3986736, upper bound: 112.5315216
time: 0.65 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A1_B2_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3997233, upper bound: 112.5333844
time: 0.74 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -9.0324421, 47.0908356, -13.6946287, 74.9554520, -83.9878922, 60.7854614
1: -14.5896549, 55.8870163, -22.2909222, 89.1237564, -103.7134018, 78.1779175
2: -12.8302002, 56.0234489, -19.4143085, 89.1291199, -101.9593201, 75.4377518
3: -24.3779488, 50.2626076, -37.6152496, 79.4628983, -103.8408508, 87.8778534
4: -18.2523689, 50.0445366, -28.1466351, 78.6116486, -96.8640137, 78.1911469

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4199636, upper bound: 112.4992420
time: 0.63 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4417061, upper bound: 112.5391282
time: 0.84 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 5.18 seconds
IS_B2_A1_A2_B1_A1_B1_A1_A1_B1_B1, status: Status.VERIFIED, split count: 10, time: 5.18
Output dim: 3, lower bound: -112.4795331, upper bound: 112.5236783
IS_B2_A1_A2_B1_A1_B1_A1_A1_B1_B2, status: Status.VERIFIED, split count: 10, time: 5.18
Output dim: 3, lower bound: -112.4558904, upper bound: 112.5078919
IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 10, time: 5.18
Output dim: 3, lower bound: -112.4742642, upper bound: 112.5467094
IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A2, status: Status.VERIFIED, split count: 10, time: 5.18
Output dim: 3, lower bound: -112.4558904, upper bound: 112.5078919
IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 10, time: 5.18
Output dim: 3, lower bound: -112.4885314, upper bound: 112.5402522
IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B2, status: Status.VERIFIED, split count: 10, time: 5.18
Output dim: 3, lower bound: -112.4674642, upper bound: 112.5316413
IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 10, time: 5.18
Output dim: 3, lower bound: -112.4851516, upper bound: 112.5664392
IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A2, status: Status.VERIFIED, split count: 10, time: 5.18
Output dim: 3, lower bound: -112.4674642, upper bound: 112.5316413
IS_B2_A1_A2_B1_A1_B2_A1_A1_B2_A1, status: Status.VERIFIED, split count: 10, time: 5.18
Output dim: 3, lower bound: -112.3986736, upper bound: 112.5315216
IS_B2_A1_A2_B1_A1_B2_A1_A1_B2_A2, status: Status.VERIFIED, split count: 10, time: 5.18
Output dim: 3, lower bound: -112.3997233, upper bound: 112.5333844
IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A1, status: Status.VERIFIED, split count: 10, time: 5.18
Output dim: 3, lower bound: -112.4199636, upper bound: 112.4992420
IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 10, time: 5.18
Output dim: 3, lower bound: -112.4417061, upper bound: 112.5391282

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -8.2466955, 42.9909363, -11.0554247, 57.3105965, -65.5572891, 54.0463600
1: -13.2549314, 50.9771156, -18.0167179, 68.1663284, -81.4212418, 68.9938354
2: -11.7257290, 51.1269493, -15.6248369, 68.3826828, -80.1084061, 66.7517853
3: -22.0498962, 45.6407623, -30.1625900, 61.4011154, -83.4510117, 75.8033524
4: -16.5148106, 45.5474091, -22.4066334, 61.1075096, -77.6223221, 67.9540253

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4409199, upper bound: 112.4918307
time: 0.70 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4409199, upper bound: 112.5466299
time: 0.68 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -11.1218109, 58.8335800, -9.0941439, 47.4433556, -58.5651665, 67.9277267
1: -18.0655937, 69.9301453, -14.6898251, 56.3069878, -74.3725815, 84.6199722
2: -15.7431889, 70.0591888, -12.9171495, 56.4523392, -72.1955185, 82.9763412
3: -30.3531590, 62.8024178, -24.5482845, 50.6305008, -80.9836426, 87.3507004
4: -22.6403103, 62.3399811, -18.3753891, 50.4139977, -73.0543060, 80.7153702

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5870653, upper bound: 112.5859072
time: 0.75 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6204332, upper bound: 112.6018683
time: 0.71 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -8.9720097, 46.8102112, -11.0554247, 57.3105965, -66.2826080, 57.8656311
1: -14.4921465, 55.5475731, -18.0167179, 68.1663284, -82.6584702, 73.5642929
2: -12.7482491, 55.6850166, -15.6248369, 68.3826828, -81.1309357, 71.3098526
3: -24.2095280, 49.9389343, -30.1625900, 61.4011154, -85.6106415, 80.1015244
4: -18.1245804, 49.7278595, -22.4066334, 61.1075096, -79.2320862, 72.1344910

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4541951, upper bound: 112.5142917
time: 0.70 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4828685, upper bound: 112.5662150
time: 0.66 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -8.8356991, 46.0971527, -13.6946287, 74.9554520, -83.7911530, 59.7917786
1: -14.2567749, 54.6923294, -22.2909222, 89.1237564, -103.3805313, 76.9832382
2: -12.5467997, 54.8283539, -19.4143085, 89.1291199, -101.6759109, 74.2426605
3: -23.8191166, 49.1714096, -37.6152496, 79.4628983, -103.2820053, 86.7866287
4: -17.8348236, 48.9676857, -28.1466351, 78.6116486, -96.4464722, 77.1142807

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3960661, upper bound: 112.5261033
time: 0.67 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3967374, upper bound: 112.5277609
time: 0.63 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 5.38 seconds
IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A1, status: Status.VERIFIED, split count: 11, time: 5.38
Output dim: 3, lower bound: -112.4409199, upper bound: 112.4918307
IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 11, time: 5.38
Output dim: 3, lower bound: -112.4409199, upper bound: 112.5466299
IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 11, time: 5.38
Output dim: 3, lower bound: -112.5870653, upper bound: 112.5859072
IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 11, time: 5.38
Output dim: 3, lower bound: -112.6204332, upper bound: 112.6018683
IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 11, time: 5.38
Output dim: 3, lower bound: -112.4541951, upper bound: 112.5142917
IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 11, time: 5.38
Output dim: 3, lower bound: -112.4828685, upper bound: 112.5662150
IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A2_A1, status: Status.VERIFIED, split count: 11, time: 5.38
Output dim: 3, lower bound: -112.3960661, upper bound: 112.5261033
IS_B2_A1_A2_B1_A1_B2_A1_A2_A1_A2_A2, status: Status.VERIFIED, split count: 11, time: 5.38
Output dim: 3, lower bound: -112.3967374, upper bound: 112.5277609

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -8.0524035, 42.0248184, -11.0554247, 57.3105965, -65.3629913, 53.0802422
1: -12.9255886, 49.8162193, -18.0167179, 68.1663284, -81.0918961, 67.8329391
2: -11.4459991, 49.9640503, -15.6248369, 68.3826828, -79.8286819, 65.5888901
3: -21.4974823, 44.5781364, -30.1625900, 61.4011154, -82.8985977, 74.7407074
4: -16.1042461, 44.4915428, -22.4066334, 61.1075096, -77.2117538, 66.8981781

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4719811, upper bound: 112.5466299
time: 0.64 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4708679, upper bound: 112.5435562
time: 0.70 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -11.1218109, 58.8335800, -8.6619520, 45.1580467, -56.2798576, 67.4955292
1: -18.0655937, 69.9301453, -13.9737263, 53.5693283, -71.6349182, 83.9038696
2: -15.7431889, 70.0591888, -12.3191652, 53.7036133, -69.4468002, 82.3783569
3: -30.3531590, 62.8024178, -23.3387222, 48.1727905, -78.5259476, 86.1411362
4: -22.6403103, 62.3399811, -17.4812431, 47.9783363, -70.6186447, 79.8212280

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5870653, upper bound: 112.5859072
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5870653, upper bound: 112.5859072
time: 0.74 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -11.1218109, 58.8335800, -8.9022999, 46.4714699, -57.5932808, 67.7358780
1: -18.0655937, 69.9301453, -14.3659439, 55.1408424, -73.2064209, 84.2960892
2: -15.7431889, 70.0591888, -12.6417618, 55.2854424, -71.0286331, 82.7009506
3: -30.3531590, 62.8024178, -24.0053177, 49.5662994, -79.9194565, 86.8077316
4: -22.6403103, 62.3399811, -17.9695358, 49.3641396, -72.0044479, 80.3095169

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6204332, upper bound: 112.6018683
time: 0.67 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6204332, upper bound: 112.6018683
time: 0.75 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -8.7800484, 45.8386688, -11.0554247, 57.3105965, -66.0906372, 56.8940887
1: -14.1677895, 54.3797188, -18.0167179, 68.1663284, -82.3341064, 72.3964310
2: -12.4728899, 54.5168915, -15.6248369, 68.3826828, -80.8555679, 70.1417236
3: -23.6653862, 48.8731003, -30.1625900, 61.4011154, -85.0664978, 79.0356903
4: -17.7185650, 48.6760750, -22.4066334, 61.1075096, -78.8260727, 71.0827026

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4828685, upper bound: 112.5662150
time: 0.65 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4796641, upper bound: 112.5600309
time: 0.67 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 5.06 seconds
IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 12, time: 5.06
Output dim: 3, lower bound: -112.4719811, upper bound: 112.5466299
IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 12, time: 5.06
Output dim: 3, lower bound: -112.4708679, upper bound: 112.5435562
IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 12, time: 5.06
Output dim: 3, lower bound: -112.5870653, upper bound: 112.5859072
IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 12, time: 5.06
Output dim: 3, lower bound: -112.5870653, upper bound: 112.5859072
IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 12, time: 5.06
Output dim: 3, lower bound: -112.6204332, upper bound: 112.6018683
IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 12, time: 5.06
Output dim: 3, lower bound: -112.6204332, upper bound: 112.6018683
IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 12, time: 5.06
Output dim: 3, lower bound: -112.4828685, upper bound: 112.5662150
IS_B2_A1_A2_B1_A1_B1_A1_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 12, time: 5.06
Output dim: 3, lower bound: -112.4796641, upper bound: 112.5600309

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: -7.4732070, 39.1394043, -11.0554247, 57.3105965, -64.7838058, 50.1948280
1: -11.9841290, 46.3424339, -18.0167179, 68.1663284, -80.1504440, 64.3591537
2: -10.6585436, 46.4686317, -15.6248369, 68.3826828, -79.0412292, 62.0934677
3: -19.8868904, 41.3877754, -30.1625900, 61.4011154, -81.2880096, 71.5503693
4: -14.9089622, 41.3214722, -22.4066334, 61.1075096, -76.0164719, 63.7281036

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4280641, upper bound: 112.5376780
time: 0.64 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 45

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 45

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 14
type: B, layer: 3, pos: 14
type: B, layer: 3, pos: 27
type: A, layer: 3, pos: 27
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 22
type: B, layer: 3, pos: 28
type: A, layer: 3, pos: 22
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 28
type: B, layer: 3, pos: 19
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 12
type: B, layer: 3, pos: 12
type: A, layer: 3, pos: 19
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 40
type: A, layer: 3, pos: 40
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 44
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 42
type: B, layer: 3, pos: 33
type: B, layer: 3, pos: 42
type: A, layer: 3, pos: 33
type: A, layer: 3, pos: 21
type: B, layer: 3, pos: 21
type: A, layer: 3, pos: 11
type: B, layer: 3, pos: 11
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 30

Time for candidate selection: 13.14 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 14

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 14

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 27

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 27

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 22

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 28

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 22

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4719811, upper bound: 112.5466299
time: 0.64 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4718829, upper bound: 112.5443100
time: 0.68 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: -7.9962268, 41.7386513, -11.0554247, 57.3105965, -65.3068161, 52.7940750
1: -12.8300686, 49.4724159, -18.0167179, 68.1663284, -80.9963837, 67.4891357
2: -11.3665257, 49.6189766, -15.6248369, 68.3826828, -79.7491989, 65.2438126
3: -21.3362980, 44.2631989, -30.1625900, 61.4011154, -82.7374115, 74.4257889
4: -15.9849434, 44.1804924, -22.4066334, 61.1075096, -77.0924530, 66.5871124

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4708679, upper bound: 112.5435562
time: 0.63 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A2_B2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4708679, upper bound: 112.5435562
time: 0.67 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -8.9720097, 46.8102112, -8.6619520, 45.1580467, -54.1300507, 55.4721642
1: -14.4921465, 55.5475731, -13.9737263, 53.5693283, -68.0614700, 69.5213013
2: -12.7482491, 55.6850166, -12.3191652, 53.7036133, -66.4518585, 68.0041656
3: -24.2095280, 49.9389343, -23.3387222, 48.1727905, -72.3823166, 73.2776489
4: -18.1245804, 49.7278595, -17.4812431, 47.9783363, -66.1029205, 67.2091064

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5862432, upper bound: 112.5851256
time: 0.68 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A1_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5870653, upper bound: 112.5859072
time: 0.64 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -10.9774284, 58.1975555, -8.6619520, 45.1580467, -56.1354713, 66.8594894
1: -17.8106995, 69.1622543, -13.9737263, 53.5693283, -71.3800278, 83.1359711
2: -15.4978085, 69.2884979, -12.3191652, 53.7036133, -69.2014236, 81.6076508
3: -29.9252090, 62.0707397, -23.3387222, 48.1727905, -78.0979996, 85.4094620
4: -22.3054218, 61.6046867, -17.4812431, 47.9783363, -70.2837524, 79.0859146

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5819579, upper bound: 112.5680271
time: 0.87 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5862432, upper bound: 112.5851256
time: 0.70 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5870653, upper bound: 112.5859072
time: 0.64 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -8.9720097, 46.8102112, -8.9022999, 46.4714699, -55.4434776, 55.7125053
1: -14.4921465, 55.5475731, -14.3659439, 55.1408424, -69.6329727, 69.9135132
2: -12.7482491, 55.6850166, -12.6417618, 55.2854424, -68.0336914, 68.3267670
3: -24.2095280, 49.9389343, -24.0053177, 49.5662994, -73.7758255, 73.9442520
4: -18.1245804, 49.7278595, -17.9695358, 49.3641396, -67.4887238, 67.6973953

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6179475, upper bound: 112.6001167
time: 0.81 seconds

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6204013, upper bound: 112.6018683
time: 0.80 seconds

## BFS IS instance: IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -10.9774284, 58.1975555, -8.9022999, 46.4714699, -57.4488983, 67.0998383
1: -17.8106995, 69.1622543, -14.3659439, 55.1408424, -72.9515305, 83.5281906
2: -15.4978085, 69.2884979, -12.6417618, 55.2854424, -70.7832489, 81.9302521
3: -29.9252090, 62.0707397, -24.0053177, 49.5662994, -79.4915085, 86.0760574
4: -22.3054218, 61.6046867, -17.9695358, 49.3641396, -71.6695557, 79.5742188

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of IS_B2_A1_A2_B1_A1_B1_A1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 5): status=Status.UNKNOWN, low=0.0742188, high=0.0761719, mid=0.0761719, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.07421875
execution time: 1130.15 seconds
