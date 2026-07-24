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
execution time: IAR + LP analysis = 2.09 + 1.73 = 3.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -112.6516518, upper bound: 112.6516518


# Binary Search by BASE starts (time budget: 1196.18 seconds, max iter: 100)

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
Binary search time: 66.38 seconds
BS Status: None
Maximum delta epsilon: None


# Individual Split (IS_dual_ind) starts
Time budget: 1129.80 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.83 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.79 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.80
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.80
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -14.8509598, 83.8702927, -96.7212601, 84.5706635
1: -20.9004955, 82.9266129, -24.1805515, 99.6739273, -120.5744247, 107.1071625
2: -18.1388531, 82.9589615, -20.9075127, 99.5991058, -117.7379608, 103.8664551
3: -35.2660179, 74.2744598, -40.9821777, 88.5371628, -123.8031769, 115.2566376
4: -26.3749428, 73.4100494, -30.6739483, 87.2144394, -113.5893555, 104.0839996

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.60 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.75 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -14.8509598, 83.8702927, -99.0249863, 100.3664780
1: -24.6804409, 101.6211853, -24.1805515, 99.6739273, -124.3543701, 125.8017349
2: -21.4360867, 101.5054321, -20.9075127, 99.5991058, -121.0351868, 122.4129333
3: -41.7919083, 90.1225281, -40.9821777, 88.5371628, -130.3290405, 131.1046753
4: -31.3080330, 88.8759460, -30.6739483, 87.2144394, -118.5224609, 119.5498962

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.66 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.37 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.37
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -12.8509674, 69.7197113, -82.5706787, 82.5706787
1: -20.9004955, 82.9266129, -20.9004955, 82.9266129, -103.8271103, 103.8271103
2: -18.1388531, 82.9589615, -18.1388531, 82.9589615, -101.0978165, 101.0978165
3: -35.2660179, 74.2744598, -35.2660179, 74.2744598, -109.5404816, 109.5404816
4: -26.3749428, 73.4100494, -26.3749428, 73.4100494, -99.7849884, 99.7849884

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.71 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -15.1547003, 85.5155258, -98.3664932, 84.8744049
1: -20.9004955, 82.9266129, -24.6804409, 101.6211853, -122.5216827, 107.6070557
2: -18.1388531, 82.9589615, -21.4360867, 101.5054321, -119.6442871, 104.3950500
3: -35.2660179, 74.2744598, -41.7919083, 90.1225281, -125.3885498, 116.0663605
4: -26.3749428, 73.4100494, -31.3080330, 88.8759460, -115.2508850, 104.7180786

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.85 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -12.8509674, 69.7197113, -84.8744049, 98.3664932
1: -24.6804409, 101.6211853, -20.9004955, 82.9266129, -107.6070557, 122.5216827
2: -21.4360867, 101.5054321, -18.1388531, 82.9589615, -104.3950500, 119.6442871
3: -41.7919083, 90.1225281, -35.2660179, 74.2744598, -116.0663605, 125.3885498
4: -31.3080330, 88.8759460, -26.3749428, 73.4100494, -104.7180786, 115.2508850

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4321057, upper bound: 112.4859769
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.74 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -15.1547003, 85.5155258, -100.6702194, 100.6702271
1: -24.6804409, 101.6211853, -24.6804409, 101.6211853, -126.3016281, 126.3016281
2: -21.4360867, 101.5054321, -21.4360867, 101.5054321, -122.9415207, 122.9415207
3: -41.7919083, 90.1225281, -41.7919083, 90.1225281, -131.9143982, 131.9143982
4: -31.3080330, 88.8759460, -31.3080330, 88.8759460, -120.1839752, 120.1839752

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4321057, upper bound: 112.4859769
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.82 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.57 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.57
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.57
Output dim: 3, lower bound: -112.4321057, upper bound: 112.4859769
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.57
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.57
Output dim: 3, lower bound: -112.4321057, upper bound: 112.4859769
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.57
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.8509674, 69.7197113, -80.1642075, 68.8153381
1: -16.9038010, 66.4907379, -20.9004955, 82.9266129, -99.8304062, 87.3912354
2: -14.7949934, 66.5433578, -18.1388531, 82.9589615, -97.7539520, 84.6822128
3: -28.4008694, 59.6408424, -35.2660179, 74.2744598, -102.6753311, 94.9068604
4: -21.3142948, 59.0830536, -26.3749428, 73.4100494, -94.7243423, 85.4579849

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.8509674, 69.7197113, -82.4571075, 82.0802765
1: -20.6976452, 82.3365021, -20.9004955, 82.9266129, -103.6242523, 103.2369919
2: -17.9375267, 82.3648682, -18.1388531, 82.9589615, -100.8964767, 100.5037231
3: -34.9264603, 73.7202835, -35.2660179, 74.2744598, -109.2009201, 108.9862976
4: -26.1118031, 72.8422928, -26.3749428, 73.4100494, -99.5218430, 99.2172318

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -15.1547003, 85.5155258, -95.9600296, 71.1190643
1: -16.9038010, 66.4907379, -24.6804409, 101.6211853, -118.5249786, 91.1711807
2: -14.7949934, 66.5433578, -21.4360867, 101.5054321, -116.3004227, 87.9794464
3: -28.4008694, 59.6408424, -41.7919083, 90.1225281, -118.5233994, 101.4327393
4: -21.3142948, 59.0830536, -31.3080330, 88.8759460, -110.1902390, 90.3910828

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5097765, upper bound: 112.4853125
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -15.1547003, 85.5155258, -98.2529221, 84.3840027
1: -20.6976452, 82.3365021, -24.6804409, 101.6211853, -122.3188324, 107.0169449
2: -17.9375267, 82.3648682, -21.4360867, 101.5054321, -119.4429474, 103.8009567
3: -34.9264603, 73.7202835, -41.7919083, 90.1225281, -125.0489883, 115.5121918
4: -26.1118031, 72.8422928, -31.3080330, 88.8759460, -114.9877472, 104.1503296

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5040140, upper bound: 112.4662794
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295
time: 0.88 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.59 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.59
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.59
Output dim: 3, lower bound: -112.5097765, upper bound: 112.4853125
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.59
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.59
Output dim: 3, lower bound: -112.5040140, upper bound: 112.4662794
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.59
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -10.4445028, 55.9643669, -66.4088669, 66.4088593
1: -16.9038010, 66.4907379, -16.9038010, 66.4907379, -83.3945312, 83.3945312
2: -14.7949934, 66.5433578, -14.7949934, 66.5433578, -81.3383484, 81.3383484
3: -28.4008694, 59.6408424, -28.4008694, 59.6408424, -88.0417099, 88.0417099
4: -21.3142948, 59.0830536, -21.3142948, 59.0830536, -80.3973465, 80.3973465

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5031682, upper bound: 112.5786717
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.7374048, 69.2293091, -79.6738129, 68.7017746
1: -16.9038010, 66.4907379, -20.6976452, 82.3365021, -99.2402802, 87.1883774
2: -14.7949934, 66.5433578, -17.9375267, 82.3648682, -97.1598587, 84.4808655
3: -28.4008694, 59.6408424, -34.9264603, 73.7202835, -102.1211548, 94.5673065
4: -21.3142948, 59.0830536, -26.1118031, 72.8422928, -94.1565857, 85.1948547

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5031682, upper bound: 112.5786717
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -10.4445028, 55.9643669, -68.7017670, 79.6738052
1: -20.6976452, 82.3365021, -16.9038010, 66.4907379, -87.1883774, 99.2402878
2: -17.9375267, 82.3648682, -14.7949934, 66.5433578, -84.4808731, 97.1598587
3: -34.9264603, 73.7202835, -28.4008694, 59.6408424, -94.5672989, 102.1211548
4: -26.1118031, 72.8422928, -21.3142948, 59.0830536, -85.1948547, 94.1565857

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4792940, upper bound: 112.5302871
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.7374048, 69.2293091, -81.9667130, 81.9667130
1: -20.6976452, 82.3365021, -20.6976452, 82.3365021, -103.0341263, 103.0341339
2: -17.9375267, 82.3648682, -17.9375267, 82.3648682, -100.3023834, 100.3023834
3: -34.9264603, 73.7202835, -34.9264603, 73.7202835, -108.6467438, 108.6467438
4: -26.1118031, 72.8422928, -26.1118031, 72.8422928, -98.9540863, 98.9540939

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4792940, upper bound: 112.5302871
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.84 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.18 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -112.5031682, upper bound: 112.5786717
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.18
Output dim: 3, lower bound: -112.5031682, upper bound: 112.5786717
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 3, lower bound: -112.4792940, upper bound: 112.5302871
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 3, lower bound: -112.4792940, upper bound: 112.5302871
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.18
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -10.4445028, 55.9643669, -66.0205383, 64.4386826
1: -16.2604694, 64.1127777, -16.9038010, 66.4907379, -82.7512054, 81.0165634
2: -14.2520180, 64.1702576, -14.7949934, 66.5433578, -80.7953568, 78.9652481
3: -27.2941570, 57.4110603, -28.4008694, 59.6408424, -86.9349976, 85.8119278
4: -20.4760780, 56.9184914, -21.3142948, 59.0830536, -79.5591202, 78.2327881

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -12.7374048, 69.2293091, -79.2854767, 66.7315750
1: -16.2604694, 64.1127777, -20.6976452, 82.3365021, -98.5969620, 84.8104095
2: -14.2520180, 64.1702576, -17.9375267, 82.3648682, -96.6168671, 82.1077805
3: -27.2941570, 57.4110603, -34.9264603, 73.7202835, -101.0144424, 92.3375092
4: -20.4760780, 56.9184914, -26.1118031, 72.8422928, -93.3183670, 83.0302811

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.58 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.92 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.92
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
Binary search (step 0): status=Status.VERIFIED, low=0.0625000, high=0.1250000, mid=0.0625000, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072838]}

## Binary search (step 1) starts
Candidate diff: 0.0937500


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.70 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.47 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.47
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.47
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -14.8509598, 83.8702927, -96.7212601, 84.5706635
1: -20.9004955, 82.9266129, -24.1805515, 99.6739273, -120.5744247, 107.1071625
2: -18.1388531, 82.9589615, -20.9075127, 99.5991058, -117.7379608, 103.8664551
3: -35.2660179, 74.2744598, -40.9821777, 88.5371628, -123.8031769, 115.2566376
4: -26.3749428, 73.4100494, -30.6739483, 87.2144394, -113.5893555, 104.0839996

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.75 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -14.8509598, 83.8702927, -99.0249863, 100.3664780
1: -24.6804409, 101.6211853, -24.1805515, 99.6739273, -124.3543701, 125.8017349
2: -21.4360867, 101.5054321, -20.9075127, 99.5991058, -121.0351868, 122.4129333
3: -41.7919083, 90.1225281, -40.9821777, 88.5371628, -130.3290405, 131.1046753
4: -31.3080330, 88.8759460, -30.6739483, 87.2144394, -118.5224609, 119.5498962

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.96 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.81 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.81
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.81
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.81
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.81
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -12.8509674, 69.7197113, -82.5706787, 82.5706787
1: -20.9004955, 82.9266129, -20.9004955, 82.9266129, -103.8271103, 103.8271103
2: -18.1388531, 82.9589615, -18.1388531, 82.9589615, -101.0978165, 101.0978165
3: -35.2660179, 74.2744598, -35.2660179, 74.2744598, -109.5404816, 109.5404816
4: -26.3749428, 73.4100494, -26.3749428, 73.4100494, -99.7849884, 99.7849884

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.72 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -15.1547003, 85.5155258, -98.3664932, 84.8744049
1: -20.9004955, 82.9266129, -24.6804409, 101.6211853, -122.5216827, 107.6070557
2: -18.1388531, 82.9589615, -21.4360867, 101.5054321, -119.6442871, 104.3950500
3: -35.2660179, 74.2744598, -41.7919083, 90.1225281, -125.3885498, 116.0663605
4: -26.3749428, 73.4100494, -31.3080330, 88.8759460, -115.2508850, 104.7180786

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.84 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -12.8509674, 69.7197113, -84.8744049, 98.3664932
1: -24.6804409, 101.6211853, -20.9004955, 82.9266129, -107.6070557, 122.5216827
2: -21.4360867, 101.5054321, -18.1388531, 82.9589615, -104.3950500, 119.6442871
3: -41.7919083, 90.1225281, -35.2660179, 74.2744598, -116.0663605, 125.3885498
4: -31.3080330, 88.8759460, -26.3749428, 73.4100494, -104.7180786, 115.2508850

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5007627
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.81 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -15.1547003, 85.5155258, -100.6702194, 100.6702271
1: -24.6804409, 101.6211853, -24.6804409, 101.6211853, -126.3016281, 126.3016281
2: -21.4360867, 101.5054321, -21.4360867, 101.5054321, -122.9415207, 122.9415207
3: -41.7919083, 90.1225281, -41.7919083, 90.1225281, -131.9143982, 131.9143982
4: -31.3080330, 88.8759460, -31.3080330, 88.8759460, -120.1839752, 120.1839752

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5007627
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.35 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.35
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5007627
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.35
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.35
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5007627
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.35
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.8509674, 69.7197113, -80.1642075, 68.8153381
1: -16.9038010, 66.4907379, -20.9004955, 82.9266129, -99.8304062, 87.3912354
2: -14.7949934, 66.5433578, -18.1388531, 82.9589615, -97.7539520, 84.6822128
3: -28.4008694, 59.6408424, -35.2660179, 74.2744598, -102.6753311, 94.9068604
4: -21.3142948, 59.0830536, -26.3749428, 73.4100494, -94.7243423, 85.4579849

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.8509674, 69.7197113, -82.4571075, 82.0802765
1: -20.6976452, 82.3365021, -20.9004955, 82.9266129, -103.6242523, 103.2369919
2: -17.9375267, 82.3648682, -18.1388531, 82.9589615, -100.8964767, 100.5037231
3: -34.9264603, 73.7202835, -35.2660179, 74.2744598, -109.2009201, 108.9862976
4: -26.1118031, 72.8422928, -26.3749428, 73.4100494, -99.5218430, 99.2172318

Time for backsubstitution: 2.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -15.1547003, 85.5155258, -95.9600296, 71.1190643
1: -16.9038010, 66.4907379, -24.6804409, 101.6211853, -118.5249786, 91.1711807
2: -14.7949934, 66.5433578, -21.4360867, 101.5054321, -116.3004227, 87.9794464
3: -28.4008694, 59.6408424, -41.7919083, 90.1225281, -118.5233994, 101.4327393
4: -21.3142948, 59.0830536, -31.3080330, 88.8759460, -110.1902390, 90.3910828

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5323850, upper bound: 112.4927724
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -15.1547003, 85.5155258, -98.2529221, 84.3840027
1: -20.6976452, 82.3365021, -24.6804409, 101.6211853, -122.3188324, 107.0169449
2: -17.9375267, 82.3648682, -21.4360867, 101.5054321, -119.4429474, 103.8009567
3: -34.9264603, 73.7202835, -41.7919083, 90.1225281, -125.0489883, 115.5121918
4: -26.1118031, 72.8422928, -31.3080330, 88.8759460, -114.9877472, 104.1503296

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5219019, upper bound: 112.4712025
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295
time: 0.83 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.71 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.71
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.71
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.71
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.71
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.71
Output dim: 3, lower bound: -112.5323850, upper bound: 112.4927724
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.71
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.71
Output dim: 3, lower bound: -112.5219019, upper bound: 112.4712025
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.71
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -10.4445028, 55.9643669, -66.4088669, 66.4088593
1: -16.9038010, 66.4907379, -16.9038010, 66.4907379, -83.3945312, 83.3945312
2: -14.7949934, 66.5433578, -14.7949934, 66.5433578, -81.3383484, 81.3383484
3: -28.4008694, 59.6408424, -28.4008694, 59.6408424, -88.0417099, 88.0417099
4: -21.3142948, 59.0830536, -21.3142948, 59.0830536, -80.3973465, 80.3973465

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5880702
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.7374048, 69.2293091, -79.6738129, 68.7017746
1: -16.9038010, 66.4907379, -20.6976452, 82.3365021, -99.2402802, 87.1883774
2: -14.7949934, 66.5433578, -17.9375267, 82.3648682, -97.1598587, 84.4808655
3: -28.4008694, 59.6408424, -34.9264603, 73.7202835, -102.1211548, 94.5673065
4: -21.3142948, 59.0830536, -26.1118031, 72.8422928, -94.1565857, 85.1948547

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5880702
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -10.4445028, 55.9643669, -68.7017670, 79.6738052
1: -20.6976452, 82.3365021, -16.9038010, 66.4907379, -87.1883774, 99.2402878
2: -17.9375267, 82.3648682, -14.7949934, 66.5433578, -84.4808731, 97.1598587
3: -34.9264603, 73.7202835, -28.4008694, 59.6408424, -94.5672989, 102.1211548
4: -26.1118031, 72.8422928, -21.3142948, 59.0830536, -85.1948547, 94.1565857

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4828694, upper bound: 112.5445520
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.7374048, 69.2293091, -81.9667130, 81.9667130
1: -20.6976452, 82.3365021, -20.6976452, 82.3365021, -103.0341263, 103.0341339
2: -17.9375267, 82.3648682, -17.9375267, 82.3648682, -100.3023834, 100.3023834
3: -34.9264603, 73.7202835, -34.9264603, 73.7202835, -108.6467438, 108.6467438
4: -26.1118031, 72.8422928, -26.1118031, 72.8422928, -98.9540863, 98.9540939

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4828694, upper bound: 112.5445520
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.21 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5880702
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5880702
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -112.4828694, upper bound: 112.5445520
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.21
Output dim: 3, lower bound: -112.4828694, upper bound: 112.5445520
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.21
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -10.4445028, 55.9643669, -66.0205383, 64.4386826
1: -16.2604694, 64.1127777, -16.9038010, 66.4907379, -82.7512054, 81.0165634
2: -14.2520180, 64.1702576, -14.7949934, 66.5433578, -80.7953568, 78.9652481
3: -27.2941570, 57.4110603, -28.4008694, 59.6408424, -86.9349976, 85.8119278
4: -20.4760780, 56.9184914, -21.3142948, 59.0830536, -79.5591202, 78.2327881

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -12.7374048, 69.2293091, -79.2854767, 66.7315750
1: -16.2604694, 64.1127777, -20.6976452, 82.3365021, -98.5969620, 84.8104095
2: -14.2520180, 64.1702576, -17.9375267, 82.3648682, -96.6168671, 82.1077805
3: -27.2941570, 57.4110603, -34.9264603, 73.7202835, -101.0144424, 92.3375092
4: -20.4760780, 56.9184914, -26.1118031, 72.8422928, -93.3183670, 83.0302811

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -10.4445028, 55.9643669, -68.2222595, 77.2292023
1: -19.9063568, 79.3842850, -16.9038010, 66.4907379, -86.3970947, 96.2880783
2: -17.2704391, 79.4186096, -14.7949934, 66.5433578, -83.8137894, 94.2136002
3: -33.5641937, 70.9576569, -28.4008694, 59.6408424, -93.2050323, 99.3585281
4: -25.0775681, 70.1679382, -21.3142948, 59.0830536, -84.1606216, 91.4822311

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -12.7374048, 69.2293091, -81.4871902, 79.5221100
1: -19.9063568, 79.3842850, -20.6976452, 82.3365021, -102.2428513, 100.0819244
2: -17.2704391, 79.4186096, -17.9375267, 82.3648682, -99.6352997, 97.3561249
3: -33.5641937, 70.9576569, -34.9264603, 73.7202835, -107.2844772, 105.8841095
4: -25.0775681, 70.1679382, -26.1118031, 72.8422928, -97.9198608, 96.2797394

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.61 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.11 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.11
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.11
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.11
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.11
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.11
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.11
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.11
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.11
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
Binary search (step 1): status=Status.VERIFIED, low=0.0937500, high=0.1250000, mid=0.0937500, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072838, 112.65165179072838]}

## Binary search (step 2) starts
Candidate diff: 0.1093750


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.69 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.62 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -14.8509598, 83.8702927, -96.7212601, 84.5706635
1: -20.9004955, 82.9266129, -24.1805515, 99.6739273, -120.5744247, 107.1071625
2: -18.1388531, 82.9589615, -20.9075127, 99.5991058, -117.7379608, 103.8664551
3: -35.2660179, 74.2744598, -40.9821777, 88.5371628, -123.8031769, 115.2566376
4: -26.3749428, 73.4100494, -30.6739483, 87.2144394, -113.5893555, 104.0839996

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.65 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -14.8509598, 83.8702927, -99.0249863, 100.3664780
1: -24.6804409, 101.6211853, -24.1805515, 99.6739273, -124.3543701, 125.8017349
2: -21.4360867, 101.5054321, -20.9075127, 99.5991058, -121.0351868, 122.4129333
3: -41.7919083, 90.1225281, -40.9821777, 88.5371628, -130.3290405, 131.1046753
4: -31.3080330, 88.8759460, -30.6739483, 87.2144394, -118.5224609, 119.5498962

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.67 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.43 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.43
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.43
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.43
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.43
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -12.8509674, 69.7197113, -82.5706787, 82.5706787
1: -20.9004955, 82.9266129, -20.9004955, 82.9266129, -103.8271103, 103.8271103
2: -18.1388531, 82.9589615, -18.1388531, 82.9589615, -101.0978165, 101.0978165
3: -35.2660179, 74.2744598, -35.2660179, 74.2744598, -109.5404816, 109.5404816
4: -26.3749428, 73.4100494, -26.3749428, 73.4100494, -99.7849884, 99.7849884

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.60 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -15.1547003, 85.5155258, -98.3664932, 84.8744049
1: -20.9004955, 82.9266129, -24.6804409, 101.6211853, -122.5216827, 107.6070557
2: -18.1388531, 82.9589615, -21.4360867, 101.5054321, -119.6442871, 104.3950500
3: -35.2660179, 74.2744598, -41.7919083, 90.1225281, -125.3885498, 116.0663605
4: -26.3749428, 73.4100494, -31.3080330, 88.8759460, -115.2508850, 104.7180786

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.82 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -12.8509674, 69.7197113, -84.8744049, 98.3664932
1: -24.6804409, 101.6211853, -20.9004955, 82.9266129, -107.6070557, 122.5216827
2: -21.4360867, 101.5054321, -18.1388531, 82.9589615, -104.3950500, 119.6442871
3: -41.7919083, 90.1225281, -35.2660179, 74.2744598, -116.0663605, 125.3885498
4: -31.3080330, 88.8759460, -26.3749428, 73.4100494, -104.7180786, 115.2508850

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.76 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -15.1547003, 85.5155258, -100.6702194, 100.6702271
1: -24.6804409, 101.6211853, -24.6804409, 101.6211853, -126.3016281, 126.3016281
2: -21.4360867, 101.5054321, -21.4360867, 101.5054321, -122.9415207, 122.9415207
3: -41.7919083, 90.1225281, -41.7919083, 90.1225281, -131.9143982, 131.9143982
4: -31.3080330, 88.8759460, -31.3080330, 88.8759460, -120.1839752, 120.1839752

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.27 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.27
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.27
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.27
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.27
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.27
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.8509674, 69.7197113, -80.1642075, 68.8153381
1: -16.9038010, 66.4907379, -20.9004955, 82.9266129, -99.8304062, 87.3912354
2: -14.7949934, 66.5433578, -18.1388531, 82.9589615, -97.7539520, 84.6822128
3: -28.4008694, 59.6408424, -35.2660179, 74.2744598, -102.6753311, 94.9068604
4: -21.3142948, 59.0830536, -26.3749428, 73.4100494, -94.7243423, 85.4579849

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.8509674, 69.7197113, -82.4571075, 82.0802765
1: -20.6976452, 82.3365021, -20.9004955, 82.9266129, -103.6242523, 103.2369919
2: -17.9375267, 82.3648682, -18.1388531, 82.9589615, -100.8964767, 100.5037231
3: -34.9264603, 73.7202835, -35.2660179, 74.2744598, -109.2009201, 108.9862976
4: -26.1118031, 72.8422928, -26.3749428, 73.4100494, -99.5218430, 99.2172318

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -15.1547003, 85.5155258, -95.9600296, 71.1190643
1: -16.9038010, 66.4907379, -24.6804409, 101.6211853, -118.5249786, 91.1711807
2: -14.7949934, 66.5433578, -21.4360867, 101.5054321, -116.3004227, 87.9794464
3: -28.4008694, 59.6408424, -41.7919083, 90.1225281, -118.5233994, 101.4327393
4: -21.3142948, 59.0830536, -31.3080330, 88.8759460, -110.1902390, 90.3910828

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5386413, upper bound: 112.4936316
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -15.1547003, 85.5155258, -98.2529221, 84.3840027
1: -20.6976452, 82.3365021, -24.6804409, 101.6211853, -122.3188324, 107.0169449
2: -17.9375267, 82.3648682, -21.4360867, 101.5054321, -119.4429474, 103.8009567
3: -34.9264603, 73.7202835, -41.7919083, 90.1225281, -125.0489883, 115.5121918
4: -26.1118031, 72.8422928, -31.3080330, 88.8759460, -114.9877472, 104.1503296

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295
time: 0.85 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.61 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.61
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.61
Output dim: 3, lower bound: -112.5386413, upper bound: 112.4936316
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.61
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.61
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.61
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -10.4445028, 55.9643669, -66.4088669, 66.4088593
1: -16.9038010, 66.4907379, -16.9038010, 66.4907379, -83.3945312, 83.3945312
2: -14.7949934, 66.5433578, -14.7949934, 66.5433578, -81.3383484, 81.3383484
3: -28.4008694, 59.6408424, -28.4008694, 59.6408424, -88.0417099, 88.0417099
4: -21.3142948, 59.0830536, -21.3142948, 59.0830536, -80.3973465, 80.3973465

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.7374048, 69.2293091, -79.6738129, 68.7017746
1: -16.9038010, 66.4907379, -20.6976452, 82.3365021, -99.2402802, 87.1883774
2: -14.7949934, 66.5433578, -17.9375267, 82.3648682, -97.1598587, 84.4808655
3: -28.4008694, 59.6408424, -34.9264603, 73.7202835, -102.1211548, 94.5673065
4: -21.3142948, 59.0830536, -26.1118031, 72.8422928, -94.1565857, 85.1948547

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -10.4445028, 55.9643669, -68.7017670, 79.6738052
1: -20.6976452, 82.3365021, -16.9038010, 66.4907379, -87.1883774, 99.2402878
2: -17.9375267, 82.3648682, -14.7949934, 66.5433578, -84.4808731, 97.1598587
3: -34.9264603, 73.7202835, -28.4008694, 59.6408424, -94.5672989, 102.1211548
4: -26.1118031, 72.8422928, -21.3142948, 59.0830536, -85.1948547, 94.1565857

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.7374048, 69.2293091, -81.9667130, 81.9667130
1: -20.6976452, 82.3365021, -20.6976452, 82.3365021, -103.0341263, 103.0341339
2: -17.9375267, 82.3648682, -17.9375267, 82.3648682, -100.3023834, 100.3023834
3: -34.9264603, 73.7202835, -34.9264603, 73.7202835, -108.6467438, 108.6467438
4: -26.1118031, 72.8422928, -26.1118031, 72.8422928, -98.9540863, 98.9540939

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.58 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.88 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -10.4445028, 55.9643669, -66.0205383, 64.4386826
1: -16.2604694, 64.1127777, -16.9038010, 66.4907379, -82.7512054, 81.0165634
2: -14.2520180, 64.1702576, -14.7949934, 66.5433578, -80.7953568, 78.9652481
3: -27.2941570, 57.4110603, -28.4008694, 59.6408424, -86.9349976, 85.8119278
4: -20.4760780, 56.9184914, -21.3142948, 59.0830536, -79.5591202, 78.2327881

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -12.7374048, 69.2293091, -79.2854767, 66.7315750
1: -16.2604694, 64.1127777, -20.6976452, 82.3365021, -98.5969620, 84.8104095
2: -14.2520180, 64.1702576, -17.9375267, 82.3648682, -96.6168671, 82.1077805
3: -27.2941570, 57.4110603, -34.9264603, 73.7202835, -101.0144424, 92.3375092
4: -20.4760780, 56.9184914, -26.1118031, 72.8422928, -93.3183670, 83.0302811

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -10.4445028, 55.9643669, -68.2222595, 77.2292023
1: -19.9063568, 79.3842850, -16.9038010, 66.4907379, -86.3970947, 96.2880783
2: -17.2704391, 79.4186096, -14.7949934, 66.5433578, -83.8137894, 94.2136002
3: -33.5641937, 70.9576569, -28.4008694, 59.6408424, -93.2050323, 99.3585281
4: -25.0775681, 70.1679382, -21.3142948, 59.0830536, -84.1606216, 91.4822311

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -12.7374048, 69.2293091, -81.4871902, 79.5221100
1: -19.9063568, 79.3842850, -20.6976452, 82.3365021, -102.2428513, 100.0819244
2: -17.2704391, 79.4186096, -17.9375267, 82.3648682, -99.6352997, 97.3561249
3: -33.5641937, 70.9576569, -34.9264603, 73.7202835, -107.2844772, 105.8841095
4: -25.0775681, 70.1679382, -26.1118031, 72.8422928, -97.9198608, 96.2797394

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.59 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.07 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
Binary search (step 2): status=Status.VERIFIED, low=0.1093750, high=0.1250000, mid=0.1093750, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072838]}

## Binary search (step 3) starts
Candidate diff: 0.1171875


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.70 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.59 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -14.8509598, 83.8702927, -96.7212601, 84.5706635
1: -20.9004955, 82.9266129, -24.1805515, 99.6739273, -120.5744247, 107.1071625
2: -18.1388531, 82.9589615, -20.9075127, 99.5991058, -117.7379608, 103.8664551
3: -35.2660179, 74.2744598, -40.9821777, 88.5371628, -123.8031769, 115.2566376
4: -26.3749428, 73.4100494, -30.6739483, 87.2144394, -113.5893555, 104.0839996

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.72 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -14.8509598, 83.8702927, -99.0249863, 100.3664780
1: -24.6804409, 101.6211853, -24.1805515, 99.6739273, -124.3543701, 125.8017349
2: -21.4360867, 101.5054321, -20.9075127, 99.5991058, -121.0351868, 122.4129333
3: -41.7919083, 90.1225281, -40.9821777, 88.5371628, -130.3290405, 131.1046753
4: -31.3080330, 88.8759460, -30.6739483, 87.2144394, -118.5224609, 119.5498962

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.75 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.55 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.55
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -12.8509674, 69.7197113, -82.5706787, 82.5706787
1: -20.9004955, 82.9266129, -20.9004955, 82.9266129, -103.8271103, 103.8271103
2: -18.1388531, 82.9589615, -18.1388531, 82.9589615, -101.0978165, 101.0978165
3: -35.2660179, 74.2744598, -35.2660179, 74.2744598, -109.5404816, 109.5404816
4: -26.3749428, 73.4100494, -26.3749428, 73.4100494, -99.7849884, 99.7849884

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.72 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -15.1547003, 85.5155258, -98.3664932, 84.8744049
1: -20.9004955, 82.9266129, -24.6804409, 101.6211853, -122.5216827, 107.6070557
2: -18.1388531, 82.9589615, -21.4360867, 101.5054321, -119.6442871, 104.3950500
3: -35.2660179, 74.2744598, -41.7919083, 90.1225281, -125.3885498, 116.0663605
4: -26.3749428, 73.4100494, -31.3080330, 88.8759460, -115.2508850, 104.7180786

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.88 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -12.8509674, 69.7197113, -84.8744049, 98.3664932
1: -24.6804409, 101.6211853, -20.9004955, 82.9266129, -107.6070557, 122.5216827
2: -21.4360867, 101.5054321, -18.1388531, 82.9589615, -104.3950500, 119.6442871
3: -41.7919083, 90.1225281, -35.2660179, 74.2744598, -116.0663605, 125.3885498
4: -31.3080330, 88.8759460, -26.3749428, 73.4100494, -104.7180786, 115.2508850

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.83 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -15.1547003, 85.5155258, -100.6702194, 100.6702271
1: -24.6804409, 101.6211853, -24.6804409, 101.6211853, -126.3016281, 126.3016281
2: -21.4360867, 101.5054321, -21.4360867, 101.5054321, -122.9415207, 122.9415207
3: -41.7919083, 90.1225281, -41.7919083, 90.1225281, -131.9143982, 131.9143982
4: -31.3080330, 88.8759460, -31.3080330, 88.8759460, -120.1839752, 120.1839752

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.37 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.8509674, 69.7197113, -80.1642075, 68.8153381
1: -16.9038010, 66.4907379, -20.9004955, 82.9266129, -99.8304062, 87.3912354
2: -14.7949934, 66.5433578, -18.1388531, 82.9589615, -97.7539520, 84.6822128
3: -28.4008694, 59.6408424, -35.2660179, 74.2744598, -102.6753311, 94.9068604
4: -21.3142948, 59.0830536, -26.3749428, 73.4100494, -94.7243423, 85.4579849

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.8509674, 69.7197113, -82.4571075, 82.0802765
1: -20.6976452, 82.3365021, -20.9004955, 82.9266129, -103.6242523, 103.2369919
2: -17.9375267, 82.3648682, -18.1388531, 82.9589615, -100.8964767, 100.5037231
3: -34.9264603, 73.7202835, -35.2660179, 74.2744598, -109.2009201, 108.9862976
4: -26.1118031, 72.8422928, -26.3749428, 73.4100494, -99.5218430, 99.2172318

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -15.1547003, 85.5155258, -95.9600296, 71.1190643
1: -16.9038010, 66.4907379, -24.6804409, 101.6211853, -118.5249786, 91.1711807
2: -14.7949934, 66.5433578, -21.4360867, 101.5054321, -116.3004227, 87.9794464
3: -28.4008694, 59.6408424, -41.7919083, 90.1225281, -118.5233994, 101.4327393
4: -21.3142948, 59.0830536, -31.3080330, 88.8759460, -110.1902390, 90.3910828

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5404619, upper bound: 112.4936470
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -15.1547003, 85.5155258, -98.2529221, 84.3840027
1: -20.6976452, 82.3365021, -24.6804409, 101.6211853, -122.3188324, 107.0169449
2: -17.9375267, 82.3648682, -21.4360867, 101.5054321, -119.4429474, 103.8009567
3: -34.9264603, 73.7202835, -41.7919083, 90.1225281, -125.0489883, 115.5121918
4: -26.1118031, 72.8422928, -31.3080330, 88.8759460, -114.9877472, 104.1503296

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295
time: 0.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.80 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.5404619, upper bound: 112.4936470
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -10.4445028, 55.9643669, -66.4088669, 66.4088593
1: -16.9038010, 66.4907379, -16.9038010, 66.4907379, -83.3945312, 83.3945312
2: -14.7949934, 66.5433578, -14.7949934, 66.5433578, -81.3383484, 81.3383484
3: -28.4008694, 59.6408424, -28.4008694, 59.6408424, -88.0417099, 88.0417099
4: -21.3142948, 59.0830536, -21.3142948, 59.0830536, -80.3973465, 80.3973465

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.7374048, 69.2293091, -79.6738129, 68.7017746
1: -16.9038010, 66.4907379, -20.6976452, 82.3365021, -99.2402802, 87.1883774
2: -14.7949934, 66.5433578, -17.9375267, 82.3648682, -97.1598587, 84.4808655
3: -28.4008694, 59.6408424, -34.9264603, 73.7202835, -102.1211548, 94.5673065
4: -21.3142948, 59.0830536, -26.1118031, 72.8422928, -94.1565857, 85.1948547

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -10.4445028, 55.9643669, -68.7017670, 79.6738052
1: -20.6976452, 82.3365021, -16.9038010, 66.4907379, -87.1883774, 99.2402878
2: -17.9375267, 82.3648682, -14.7949934, 66.5433578, -84.4808731, 97.1598587
3: -34.9264603, 73.7202835, -28.4008694, 59.6408424, -94.5672989, 102.1211548
4: -26.1118031, 72.8422928, -21.3142948, 59.0830536, -85.1948547, 94.1565857

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.7374048, 69.2293091, -81.9667130, 81.9667130
1: -20.6976452, 82.3365021, -20.6976452, 82.3365021, -103.0341263, 103.0341339
2: -17.9375267, 82.3648682, -17.9375267, 82.3648682, -100.3023834, 100.3023834
3: -34.9264603, 73.7202835, -34.9264603, 73.7202835, -108.6467438, 108.6467438
4: -26.1118031, 72.8422928, -26.1118031, 72.8422928, -98.9540863, 98.9540939

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -14.6269550, 82.7033157, -93.1478195, 70.5913239
1: -16.9038010, 66.4907379, -23.8081474, 98.2509613, -115.1547546, 90.2988892
2: -14.7949934, 66.5433578, -20.6988335, 98.1297531, -112.9247437, 87.2421646
3: -28.4008694, 59.6408424, -40.2802467, 87.0169601, -115.4178314, 99.9210739
4: -21.3142948, 59.0830536, -30.1702652, 85.8516464, -107.1659393, 89.2533188

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4962493, upper bound: 112.4732156
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.82 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.82
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.82
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.82
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.82
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.82
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.82
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.82
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.82
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.82
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.82
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -10.4445028, 55.9643669, -66.0205383, 64.4386826
1: -16.2604694, 64.1127777, -16.9038010, 66.4907379, -82.7512054, 81.0165634
2: -14.2520180, 64.1702576, -14.7949934, 66.5433578, -80.7953568, 78.9652481
3: -27.2941570, 57.4110603, -28.4008694, 59.6408424, -86.9349976, 85.8119278
4: -20.4760780, 56.9184914, -21.3142948, 59.0830536, -79.5591202, 78.2327881

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -12.7374048, 69.2293091, -79.2854767, 66.7315750
1: -16.2604694, 64.1127777, -20.6976452, 82.3365021, -98.5969620, 84.8104095
2: -14.2520180, 64.1702576, -17.9375267, 82.3648682, -96.6168671, 82.1077805
3: -27.2941570, 57.4110603, -34.9264603, 73.7202835, -101.0144424, 92.3375092
4: -20.4760780, 56.9184914, -26.1118031, 72.8422928, -93.3183670, 83.0302811

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -10.4445028, 55.9643669, -68.2222595, 77.2292023
1: -19.9063568, 79.3842850, -16.9038010, 66.4907379, -86.3970947, 96.2880783
2: -17.2704391, 79.4186096, -14.7949934, 66.5433578, -83.8137894, 94.2136002
3: -33.5641937, 70.9576569, -28.4008694, 59.6408424, -93.2050323, 99.3585281
4: -25.0775681, 70.1679382, -21.3142948, 59.0830536, -84.1606216, 91.4822311

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -12.7374048, 69.2293091, -81.4871902, 79.5221100
1: -19.9063568, 79.3842850, -20.6976452, 82.3365021, -102.2428513, 100.0819244
2: -17.2704391, 79.4186096, -17.9375267, 82.3648682, -99.6352997, 97.3561249
3: -33.5641937, 70.9576569, -34.9264603, 73.7202835, -107.2844772, 105.8841095
4: -25.0775681, 70.1679382, -26.1118031, 72.8422928, -97.9198608, 96.2797394

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.60 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.83 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.83
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
Binary search (step 3): status=Status.VERIFIED, low=0.1171875, high=0.1250000, mid=0.1171875, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072838, 112.65165179072841]}

## Binary search (step 4) starts
Candidate diff: 0.1210938


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.81 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.70 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -14.8509598, 83.8702927, -96.7212601, 84.5706635
1: -20.9004955, 82.9266129, -24.1805515, 99.6739273, -120.5744247, 107.1071625
2: -18.1388531, 82.9589615, -20.9075127, 99.5991058, -117.7379608, 103.8664551
3: -35.2660179, 74.2744598, -40.9821777, 88.5371628, -123.8031769, 115.2566376
4: -26.3749428, 73.4100494, -30.6739483, 87.2144394, -113.5893555, 104.0839996

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.61 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -14.8509598, 83.8702927, -99.0249863, 100.3664780
1: -24.6804409, 101.6211853, -24.1805515, 99.6739273, -124.3543701, 125.8017349
2: -21.4360867, 101.5054321, -20.9075127, 99.5991058, -121.0351868, 122.4129333
3: -41.7919083, 90.1225281, -40.9821777, 88.5371628, -130.3290405, 131.1046753
4: -31.3080330, 88.8759460, -30.6739483, 87.2144394, -118.5224609, 119.5498962

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.69 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.52 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.52
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -12.8509674, 69.7197113, -82.5706787, 82.5706787
1: -20.9004955, 82.9266129, -20.9004955, 82.9266129, -103.8271103, 103.8271103
2: -18.1388531, 82.9589615, -18.1388531, 82.9589615, -101.0978165, 101.0978165
3: -35.2660179, 74.2744598, -35.2660179, 74.2744598, -109.5404816, 109.5404816
4: -26.3749428, 73.4100494, -26.3749428, 73.4100494, -99.7849884, 99.7849884

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.74 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -15.1547003, 85.5155258, -98.3664932, 84.8744049
1: -20.9004955, 82.9266129, -24.6804409, 101.6211853, -122.5216827, 107.6070557
2: -18.1388531, 82.9589615, -21.4360867, 101.5054321, -119.6442871, 104.3950500
3: -35.2660179, 74.2744598, -41.7919083, 90.1225281, -125.3885498, 116.0663605
4: -26.3749428, 73.4100494, -31.3080330, 88.8759460, -115.2508850, 104.7180786

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.94 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -12.8509674, 69.7197113, -84.8744049, 98.3664932
1: -24.6804409, 101.6211853, -20.9004955, 82.9266129, -107.6070557, 122.5216827
2: -21.4360867, 101.5054321, -18.1388531, 82.9589615, -104.3950500, 119.6442871
3: -41.7919083, 90.1225281, -35.2660179, 74.2744598, -116.0663605, 125.3885498
4: -31.3080330, 88.8759460, -26.3749428, 73.4100494, -104.7180786, 115.2508850

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.78 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -15.1547003, 85.5155258, -100.6702194, 100.6702271
1: -24.6804409, 101.6211853, -24.6804409, 101.6211853, -126.3016281, 126.3016281
2: -21.4360867, 101.5054321, -21.4360867, 101.5054321, -122.9415207, 122.9415207
3: -41.7919083, 90.1225281, -41.7919083, 90.1225281, -131.9143982, 131.9143982
4: -31.3080330, 88.8759460, -31.3080330, 88.8759460, -120.1839752, 120.1839752

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.25 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.25
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.25
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.25
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.25
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.25
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.25
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.25
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.25
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.8509674, 69.7197113, -80.1642075, 68.8153381
1: -16.9038010, 66.4907379, -20.9004955, 82.9266129, -99.8304062, 87.3912354
2: -14.7949934, 66.5433578, -18.1388531, 82.9589615, -97.7539520, 84.6822128
3: -28.4008694, 59.6408424, -35.2660179, 74.2744598, -102.6753311, 94.9068604
4: -21.3142948, 59.0830536, -26.3749428, 73.4100494, -94.7243423, 85.4579849

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.8509674, 69.7197113, -82.4571075, 82.0802765
1: -20.6976452, 82.3365021, -20.9004955, 82.9266129, -103.6242523, 103.2369919
2: -17.9375267, 82.3648682, -18.1388531, 82.9589615, -100.8964767, 100.5037231
3: -34.9264603, 73.7202835, -35.2660179, 74.2744598, -109.2009201, 108.9862976
4: -26.1118031, 72.8422928, -26.3749428, 73.4100494, -99.5218430, 99.2172318

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -15.1547003, 85.5155258, -95.9600296, 71.1190643
1: -16.9038010, 66.4907379, -24.6804409, 101.6211853, -118.5249786, 91.1711807
2: -14.7949934, 66.5433578, -21.4360867, 101.5054321, -116.3004227, 87.9794464
3: -28.4008694, 59.6408424, -41.7919083, 90.1225281, -118.5233994, 101.4327393
4: -21.3142948, 59.0830536, -31.3080330, 88.8759460, -110.1902390, 90.3910828

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5404892, upper bound: 112.4936470
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -15.1547003, 85.5155258, -98.2529221, 84.3840027
1: -20.6976452, 82.3365021, -24.6804409, 101.6211853, -122.3188324, 107.0169449
2: -17.9375267, 82.3648682, -21.4360867, 101.5054321, -119.4429474, 103.8009567
3: -34.9264603, 73.7202835, -41.7919083, 90.1225281, -125.0489883, 115.5121918
4: -26.1118031, 72.8422928, -31.3080330, 88.8759460, -114.9877472, 104.1503296

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295
time: 0.65 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.43 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 3, lower bound: -112.5404892, upper bound: 112.4936470
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.43
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -10.4445028, 55.9643669, -66.4088669, 66.4088593
1: -16.9038010, 66.4907379, -16.9038010, 66.4907379, -83.3945312, 83.3945312
2: -14.7949934, 66.5433578, -14.7949934, 66.5433578, -81.3383484, 81.3383484
3: -28.4008694, 59.6408424, -28.4008694, 59.6408424, -88.0417099, 88.0417099
4: -21.3142948, 59.0830536, -21.3142948, 59.0830536, -80.3973465, 80.3973465

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.7374048, 69.2293091, -79.6738129, 68.7017746
1: -16.9038010, 66.4907379, -20.6976452, 82.3365021, -99.2402802, 87.1883774
2: -14.7949934, 66.5433578, -17.9375267, 82.3648682, -97.1598587, 84.4808655
3: -28.4008694, 59.6408424, -34.9264603, 73.7202835, -102.1211548, 94.5673065
4: -21.3142948, 59.0830536, -26.1118031, 72.8422928, -94.1565857, 85.1948547

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -10.4445028, 55.9643669, -68.7017670, 79.6738052
1: -20.6976452, 82.3365021, -16.9038010, 66.4907379, -87.1883774, 99.2402878
2: -17.9375267, 82.3648682, -14.7949934, 66.5433578, -84.4808731, 97.1598587
3: -34.9264603, 73.7202835, -28.4008694, 59.6408424, -94.5672989, 102.1211548
4: -26.1118031, 72.8422928, -21.3142948, 59.0830536, -85.1948547, 94.1565857

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.7374048, 69.2293091, -81.9667130, 81.9667130
1: -20.6976452, 82.3365021, -20.6976452, 82.3365021, -103.0341263, 103.0341339
2: -17.9375267, 82.3648682, -17.9375267, 82.3648682, -100.3023834, 100.3023834
3: -34.9264603, 73.7202835, -34.9264603, 73.7202835, -108.6467438, 108.6467438
4: -26.1118031, 72.8422928, -26.1118031, 72.8422928, -98.9540863, 98.9540939

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -14.6269550, 82.7033157, -93.1478195, 70.5913239
1: -16.9038010, 66.4907379, -23.8081474, 98.2509613, -115.1547546, 90.2988892
2: -14.7949934, 66.5433578, -20.6988335, 98.1297531, -112.9247437, 87.2421646
3: -28.4008694, 59.6408424, -40.2802467, 87.0169601, -115.4178314, 99.9210739
4: -21.3142948, 59.0830536, -30.1702652, 85.8516464, -107.1659393, 89.2533188

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4962782, upper bound: 112.4732156
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.63 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.69 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.69
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.69
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -10.4445028, 55.9643669, -66.0205383, 64.4386826
1: -16.2604694, 64.1127777, -16.9038010, 66.4907379, -82.7512054, 81.0165634
2: -14.2520180, 64.1702576, -14.7949934, 66.5433578, -80.7953568, 78.9652481
3: -27.2941570, 57.4110603, -28.4008694, 59.6408424, -86.9349976, 85.8119278
4: -20.4760780, 56.9184914, -21.3142948, 59.0830536, -79.5591202, 78.2327881

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -12.7374048, 69.2293091, -79.2854767, 66.7315750
1: -16.2604694, 64.1127777, -20.6976452, 82.3365021, -98.5969620, 84.8104095
2: -14.2520180, 64.1702576, -17.9375267, 82.3648682, -96.6168671, 82.1077805
3: -27.2941570, 57.4110603, -34.9264603, 73.7202835, -101.0144424, 92.3375092
4: -20.4760780, 56.9184914, -26.1118031, 72.8422928, -93.3183670, 83.0302811

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -10.4445028, 55.9643669, -68.2222595, 77.2292023
1: -19.9063568, 79.3842850, -16.9038010, 66.4907379, -86.3970947, 96.2880783
2: -17.2704391, 79.4186096, -14.7949934, 66.5433578, -83.8137894, 94.2136002
3: -33.5641937, 70.9576569, -28.4008694, 59.6408424, -93.2050323, 99.3585281
4: -25.0775681, 70.1679382, -21.3142948, 59.0830536, -84.1606216, 91.4822311

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -12.7374048, 69.2293091, -81.4871902, 79.5221100
1: -19.9063568, 79.3842850, -20.6976452, 82.3365021, -102.2428513, 100.0819244
2: -17.2704391, 79.4186096, -17.9375267, 82.3648682, -99.6352997, 97.3561249
3: -33.5641937, 70.9576569, -34.9264603, 73.7202835, -107.2844772, 105.8841095
4: -25.0775681, 70.1679382, -26.1118031, 72.8422928, -97.9198608, 96.2797394

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.08 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.08
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
Binary search (step 4): status=Status.VERIFIED, low=0.1210938, high=0.1250000, mid=0.1210938, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072838, 112.65165179072841]}

## Binary search (step 5) starts
Candidate diff: 0.1230469


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.92 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.89 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.89
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.89
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -14.8509598, 83.8702927, -96.7212601, 84.5706635
1: -20.9004955, 82.9266129, -24.1805515, 99.6739273, -120.5744247, 107.1071625
2: -18.1388531, 82.9589615, -20.9075127, 99.5991058, -117.7379608, 103.8664551
3: -35.2660179, 74.2744598, -40.9821777, 88.5371628, -123.8031769, 115.2566376
4: -26.3749428, 73.4100494, -30.6739483, 87.2144394, -113.5893555, 104.0839996

Time for backsubstitution: 2.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.72 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -14.8509598, 83.8702927, -99.0249863, 100.3664780
1: -24.6804409, 101.6211853, -24.1805515, 99.6739273, -124.3543701, 125.8017349
2: -21.4360867, 101.5054321, -20.9075127, 99.5991058, -121.0351868, 122.4129333
3: -41.7919083, 90.1225281, -40.9821777, 88.5371628, -130.3290405, 131.1046753
4: -31.3080330, 88.8759460, -30.6739483, 87.2144394, -118.5224609, 119.5498962

Time for backsubstitution: 2.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.72 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.06 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.06
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -12.8509674, 69.7197113, -82.5706787, 82.5706787
1: -20.9004955, 82.9266129, -20.9004955, 82.9266129, -103.8271103, 103.8271103
2: -18.1388531, 82.9589615, -18.1388531, 82.9589615, -101.0978165, 101.0978165
3: -35.2660179, 74.2744598, -35.2660179, 74.2744598, -109.5404816, 109.5404816
4: -26.3749428, 73.4100494, -26.3749428, 73.4100494, -99.7849884, 99.7849884

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.72 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -15.1547003, 85.5155258, -98.3664932, 84.8744049
1: -20.9004955, 82.9266129, -24.6804409, 101.6211853, -122.5216827, 107.6070557
2: -18.1388531, 82.9589615, -21.4360867, 101.5054321, -119.6442871, 104.3950500
3: -35.2660179, 74.2744598, -41.7919083, 90.1225281, -125.3885498, 116.0663605
4: -26.3749428, 73.4100494, -31.3080330, 88.8759460, -115.2508850, 104.7180786

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.79 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -12.8509674, 69.7197113, -84.8744049, 98.3664932
1: -24.6804409, 101.6211853, -20.9004955, 82.9266129, -107.6070557, 122.5216827
2: -21.4360867, 101.5054321, -18.1388531, 82.9589615, -104.3950500, 119.6442871
3: -41.7919083, 90.1225281, -35.2660179, 74.2744598, -116.0663605, 125.3885498
4: -31.3080330, 88.8759460, -26.3749428, 73.4100494, -104.7180786, 115.2508850

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.80 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -15.1547003, 85.5155258, -100.6702194, 100.6702271
1: -24.6804409, 101.6211853, -24.6804409, 101.6211853, -126.3016281, 126.3016281
2: -21.4360867, 101.5054321, -21.4360867, 101.5054321, -122.9415207, 122.9415207
3: -41.7919083, 90.1225281, -41.7919083, 90.1225281, -131.9143982, 131.9143982
4: -31.3080330, 88.8759460, -31.3080330, 88.8759460, -120.1839752, 120.1839752

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.56 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.56
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.56
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.56
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.56
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.56
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.8509674, 69.7197113, -80.1642075, 68.8153381
1: -16.9038010, 66.4907379, -20.9004955, 82.9266129, -99.8304062, 87.3912354
2: -14.7949934, 66.5433578, -18.1388531, 82.9589615, -97.7539520, 84.6822128
3: -28.4008694, 59.6408424, -35.2660179, 74.2744598, -102.6753311, 94.9068604
4: -21.3142948, 59.0830536, -26.3749428, 73.4100494, -94.7243423, 85.4579849

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.8509674, 69.7197113, -82.4571075, 82.0802765
1: -20.6976452, 82.3365021, -20.9004955, 82.9266129, -103.6242523, 103.2369919
2: -17.9375267, 82.3648682, -18.1388531, 82.9589615, -100.8964767, 100.5037231
3: -34.9264603, 73.7202835, -35.2660179, 74.2744598, -109.2009201, 108.9862976
4: -26.1118031, 72.8422928, -26.3749428, 73.4100494, -99.5218430, 99.2172318

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -15.1547003, 85.5155258, -95.9600296, 71.1190643
1: -16.9038010, 66.4907379, -24.6804409, 101.6211853, -118.5249786, 91.1711807
2: -14.7949934, 66.5433578, -21.4360867, 101.5054321, -116.3004227, 87.9794464
3: -28.4008694, 59.6408424, -41.7919083, 90.1225281, -118.5233994, 101.4327393
4: -21.3142948, 59.0830536, -31.3080330, 88.8759460, -110.1902390, 90.3910828

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5404973, upper bound: 112.4936470
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -15.1547003, 85.5155258, -98.2529221, 84.3840027
1: -20.6976452, 82.3365021, -24.6804409, 101.6211853, -122.3188324, 107.0169449
2: -17.9375267, 82.3648682, -21.4360867, 101.5054321, -119.4429474, 103.8009567
3: -34.9264603, 73.7202835, -41.7919083, 90.1225281, -125.0489883, 115.5121918
4: -26.1118031, 72.8422928, -31.3080330, 88.8759460, -114.9877472, 104.1503296

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295
time: 0.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.57 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.57
Output dim: 3, lower bound: -112.5404973, upper bound: 112.4936470
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.57
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.57
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.57
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -10.4445028, 55.9643669, -66.4088669, 66.4088593
1: -16.9038010, 66.4907379, -16.9038010, 66.4907379, -83.3945312, 83.3945312
2: -14.7949934, 66.5433578, -14.7949934, 66.5433578, -81.3383484, 81.3383484
3: -28.4008694, 59.6408424, -28.4008694, 59.6408424, -88.0417099, 88.0417099
4: -21.3142948, 59.0830536, -21.3142948, 59.0830536, -80.3973465, 80.3973465

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.7374048, 69.2293091, -79.6738129, 68.7017746
1: -16.9038010, 66.4907379, -20.6976452, 82.3365021, -99.2402802, 87.1883774
2: -14.7949934, 66.5433578, -17.9375267, 82.3648682, -97.1598587, 84.4808655
3: -28.4008694, 59.6408424, -34.9264603, 73.7202835, -102.1211548, 94.5673065
4: -21.3142948, 59.0830536, -26.1118031, 72.8422928, -94.1565857, 85.1948547

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -10.4445028, 55.9643669, -68.7017670, 79.6738052
1: -20.6976452, 82.3365021, -16.9038010, 66.4907379, -87.1883774, 99.2402878
2: -17.9375267, 82.3648682, -14.7949934, 66.5433578, -84.4808731, 97.1598587
3: -34.9264603, 73.7202835, -28.4008694, 59.6408424, -94.5672989, 102.1211548
4: -26.1118031, 72.8422928, -21.3142948, 59.0830536, -85.1948547, 94.1565857

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.7374048, 69.2293091, -81.9667130, 81.9667130
1: -20.6976452, 82.3365021, -20.6976452, 82.3365021, -103.0341263, 103.0341339
2: -17.9375267, 82.3648682, -17.9375267, 82.3648682, -100.3023834, 100.3023834
3: -34.9264603, 73.7202835, -34.9264603, 73.7202835, -108.6467438, 108.6467438
4: -26.1118031, 72.8422928, -26.1118031, 72.8422928, -98.9540863, 98.9540939

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -14.6269550, 82.7033157, -93.1478195, 70.5913239
1: -16.9038010, 66.4907379, -23.8081474, 98.2509613, -115.1547546, 90.2988892
2: -14.7949934, 66.5433578, -20.6988335, 98.1297531, -112.9247437, 87.2421646
3: -28.4008694, 59.6408424, -40.2802467, 87.0169601, -115.4178314, 99.9210739
4: -21.3142948, 59.0830536, -30.1702652, 85.8516464, -107.1659393, 89.2533188

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4962782, upper bound: 112.4732156
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.61 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.71 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.71
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.71
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -10.4445028, 55.9643669, -66.0205383, 64.4386826
1: -16.2604694, 64.1127777, -16.9038010, 66.4907379, -82.7512054, 81.0165634
2: -14.2520180, 64.1702576, -14.7949934, 66.5433578, -80.7953568, 78.9652481
3: -27.2941570, 57.4110603, -28.4008694, 59.6408424, -86.9349976, 85.8119278
4: -20.4760780, 56.9184914, -21.3142948, 59.0830536, -79.5591202, 78.2327881

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -12.7374048, 69.2293091, -79.2854767, 66.7315750
1: -16.2604694, 64.1127777, -20.6976452, 82.3365021, -98.5969620, 84.8104095
2: -14.2520180, 64.1702576, -17.9375267, 82.3648682, -96.6168671, 82.1077805
3: -27.2941570, 57.4110603, -34.9264603, 73.7202835, -101.0144424, 92.3375092
4: -20.4760780, 56.9184914, -26.1118031, 72.8422928, -93.3183670, 83.0302811

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -10.4445028, 55.9643669, -68.2222595, 77.2292023
1: -19.9063568, 79.3842850, -16.9038010, 66.4907379, -86.3970947, 96.2880783
2: -17.2704391, 79.4186096, -14.7949934, 66.5433578, -83.8137894, 94.2136002
3: -33.5641937, 70.9576569, -28.4008694, 59.6408424, -93.2050323, 99.3585281
4: -25.0775681, 70.1679382, -21.3142948, 59.0830536, -84.1606216, 91.4822311

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -12.7374048, 69.2293091, -81.4871902, 79.5221100
1: -19.9063568, 79.3842850, -20.6976452, 82.3365021, -102.2428513, 100.0819244
2: -17.2704391, 79.4186096, -17.9375267, 82.3648682, -99.6352997, 97.3561249
3: -33.5641937, 70.9576569, -34.9264603, 73.7202835, -107.2844772, 105.8841095
4: -25.0775681, 70.1679382, -26.1118031, 72.8422928, -97.9198608, 96.2797394

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.57 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.15 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.15
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.15
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.15
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.15
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.15
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.15
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.15
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.15
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
Binary search (step 5): status=Status.VERIFIED, low=0.1230469, high=0.1250000, mid=0.1230469, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 6) starts
Candidate diff: 0.1240234


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.57 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -14.8509598, 83.8702927, -96.7212601, 84.5706635
1: -20.9004955, 82.9266129, -24.1805515, 99.6739273, -120.5744247, 107.1071625
2: -18.1388531, 82.9589615, -20.9075127, 99.5991058, -117.7379608, 103.8664551
3: -35.2660179, 74.2744598, -40.9821777, 88.5371628, -123.8031769, 115.2566376
4: -26.3749428, 73.4100494, -30.6739483, 87.2144394, -113.5893555, 104.0839996

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.70 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -14.8509598, 83.8702927, -99.0249863, 100.3664780
1: -24.6804409, 101.6211853, -24.1805515, 99.6739273, -124.3543701, 125.8017349
2: -21.4360867, 101.5054321, -20.9075127, 99.5991058, -121.0351868, 122.4129333
3: -41.7919083, 90.1225281, -40.9821777, 88.5371628, -130.3290405, 131.1046753
4: -31.3080330, 88.8759460, -30.6739483, 87.2144394, -118.5224609, 119.5498962

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.76 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.98 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.98
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -12.8509674, 69.7197113, -82.5706787, 82.5706787
1: -20.9004955, 82.9266129, -20.9004955, 82.9266129, -103.8271103, 103.8271103
2: -18.1388531, 82.9589615, -18.1388531, 82.9589615, -101.0978165, 101.0978165
3: -35.2660179, 74.2744598, -35.2660179, 74.2744598, -109.5404816, 109.5404816
4: -26.3749428, 73.4100494, -26.3749428, 73.4100494, -99.7849884, 99.7849884

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.72 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -15.1547003, 85.5155258, -98.3664932, 84.8744049
1: -20.9004955, 82.9266129, -24.6804409, 101.6211853, -122.5216827, 107.6070557
2: -18.1388531, 82.9589615, -21.4360867, 101.5054321, -119.6442871, 104.3950500
3: -35.2660179, 74.2744598, -41.7919083, 90.1225281, -125.3885498, 116.0663605
4: -26.3749428, 73.4100494, -31.3080330, 88.8759460, -115.2508850, 104.7180786

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.70 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -12.8509674, 69.7197113, -84.8744049, 98.3664932
1: -24.6804409, 101.6211853, -20.9004955, 82.9266129, -107.6070557, 122.5216827
2: -21.4360867, 101.5054321, -18.1388531, 82.9589615, -104.3950500, 119.6442871
3: -41.7919083, 90.1225281, -35.2660179, 74.2744598, -116.0663605, 125.3885498
4: -31.3080330, 88.8759460, -26.3749428, 73.4100494, -104.7180786, 115.2508850

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.83 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -15.1547003, 85.5155258, -100.6702194, 100.6702271
1: -24.6804409, 101.6211853, -24.6804409, 101.6211853, -126.3016281, 126.3016281
2: -21.4360867, 101.5054321, -21.4360867, 101.5054321, -122.9415207, 122.9415207
3: -41.7919083, 90.1225281, -41.7919083, 90.1225281, -131.9143982, 131.9143982
4: -31.3080330, 88.8759460, -31.3080330, 88.8759460, -120.1839752, 120.1839752

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.37 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.37
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.8509674, 69.7197113, -80.1642075, 68.8153381
1: -16.9038010, 66.4907379, -20.9004955, 82.9266129, -99.8304062, 87.3912354
2: -14.7949934, 66.5433578, -18.1388531, 82.9589615, -97.7539520, 84.6822128
3: -28.4008694, 59.6408424, -35.2660179, 74.2744598, -102.6753311, 94.9068604
4: -21.3142948, 59.0830536, -26.3749428, 73.4100494, -94.7243423, 85.4579849

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.8509674, 69.7197113, -82.4571075, 82.0802765
1: -20.6976452, 82.3365021, -20.9004955, 82.9266129, -103.6242523, 103.2369919
2: -17.9375267, 82.3648682, -18.1388531, 82.9589615, -100.8964767, 100.5037231
3: -34.9264603, 73.7202835, -35.2660179, 74.2744598, -109.2009201, 108.9862976
4: -26.1118031, 72.8422928, -26.3749428, 73.4100494, -99.5218430, 99.2172318

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -15.1547003, 85.5155258, -95.9600296, 71.1190643
1: -16.9038010, 66.4907379, -24.6804409, 101.6211853, -118.5249786, 91.1711807
2: -14.7949934, 66.5433578, -21.4360867, 101.5054321, -116.3004227, 87.9794464
3: -28.4008694, 59.6408424, -41.7919083, 90.1225281, -118.5233994, 101.4327393
4: -21.3142948, 59.0830536, -31.3080330, 88.8759460, -110.1902390, 90.3910828

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5405009, upper bound: 112.4936470
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -15.1547003, 85.5155258, -98.2529221, 84.3840027
1: -20.6976452, 82.3365021, -24.6804409, 101.6211853, -122.3188324, 107.0169449
2: -17.9375267, 82.3648682, -21.4360867, 101.5054321, -119.4429474, 103.8009567
3: -34.9264603, 73.7202835, -41.7919083, 90.1225281, -125.0489883, 115.5121918
4: -26.1118031, 72.8422928, -31.3080330, 88.8759460, -114.9877472, 104.1503296

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295
time: 0.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.54 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.54
Output dim: 3, lower bound: -112.5405009, upper bound: 112.4936470
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.54
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -10.4445028, 55.9643669, -66.4088669, 66.4088593
1: -16.9038010, 66.4907379, -16.9038010, 66.4907379, -83.3945312, 83.3945312
2: -14.7949934, 66.5433578, -14.7949934, 66.5433578, -81.3383484, 81.3383484
3: -28.4008694, 59.6408424, -28.4008694, 59.6408424, -88.0417099, 88.0417099
4: -21.3142948, 59.0830536, -21.3142948, 59.0830536, -80.3973465, 80.3973465

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.7374048, 69.2293091, -79.6738129, 68.7017746
1: -16.9038010, 66.4907379, -20.6976452, 82.3365021, -99.2402802, 87.1883774
2: -14.7949934, 66.5433578, -17.9375267, 82.3648682, -97.1598587, 84.4808655
3: -28.4008694, 59.6408424, -34.9264603, 73.7202835, -102.1211548, 94.5673065
4: -21.3142948, 59.0830536, -26.1118031, 72.8422928, -94.1565857, 85.1948547

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -10.4445028, 55.9643669, -68.7017670, 79.6738052
1: -20.6976452, 82.3365021, -16.9038010, 66.4907379, -87.1883774, 99.2402878
2: -17.9375267, 82.3648682, -14.7949934, 66.5433578, -84.4808731, 97.1598587
3: -34.9264603, 73.7202835, -28.4008694, 59.6408424, -94.5672989, 102.1211548
4: -26.1118031, 72.8422928, -21.3142948, 59.0830536, -85.1948547, 94.1565857

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.7374048, 69.2293091, -81.9667130, 81.9667130
1: -20.6976452, 82.3365021, -20.6976452, 82.3365021, -103.0341263, 103.0341339
2: -17.9375267, 82.3648682, -17.9375267, 82.3648682, -100.3023834, 100.3023834
3: -34.9264603, 73.7202835, -34.9264603, 73.7202835, -108.6467438, 108.6467438
4: -26.1118031, 72.8422928, -26.1118031, 72.8422928, -98.9540863, 98.9540939

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -14.6269550, 82.7033157, -93.1478195, 70.5913239
1: -16.9038010, 66.4907379, -23.8081474, 98.2509613, -115.1547546, 90.2988892
2: -14.7949934, 66.5433578, -20.6988335, 98.1297531, -112.9247437, 87.2421646
3: -28.4008694, 59.6408424, -40.2802467, 87.0169601, -115.4178314, 99.9210739
4: -21.3142948, 59.0830536, -30.1702652, 85.8516464, -107.1659393, 89.2533188

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4962782, upper bound: 112.4732156
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.67 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.70 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.70
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -10.4445028, 55.9643669, -66.0205383, 64.4386826
1: -16.2604694, 64.1127777, -16.9038010, 66.4907379, -82.7512054, 81.0165634
2: -14.2520180, 64.1702576, -14.7949934, 66.5433578, -80.7953568, 78.9652481
3: -27.2941570, 57.4110603, -28.4008694, 59.6408424, -86.9349976, 85.8119278
4: -20.4760780, 56.9184914, -21.3142948, 59.0830536, -79.5591202, 78.2327881

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -12.7374048, 69.2293091, -79.2854767, 66.7315750
1: -16.2604694, 64.1127777, -20.6976452, 82.3365021, -98.5969620, 84.8104095
2: -14.2520180, 64.1702576, -17.9375267, 82.3648682, -96.6168671, 82.1077805
3: -27.2941570, 57.4110603, -34.9264603, 73.7202835, -101.0144424, 92.3375092
4: -20.4760780, 56.9184914, -26.1118031, 72.8422928, -93.3183670, 83.0302811

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -10.4445028, 55.9643669, -68.2222595, 77.2292023
1: -19.9063568, 79.3842850, -16.9038010, 66.4907379, -86.3970947, 96.2880783
2: -17.2704391, 79.4186096, -14.7949934, 66.5433578, -83.8137894, 94.2136002
3: -33.5641937, 70.9576569, -28.4008694, 59.6408424, -93.2050323, 99.3585281
4: -25.0775681, 70.1679382, -21.3142948, 59.0830536, -84.1606216, 91.4822311

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -12.7374048, 69.2293091, -81.4871902, 79.5221100
1: -19.9063568, 79.3842850, -20.6976452, 82.3365021, -102.2428513, 100.0819244
2: -17.2704391, 79.4186096, -17.9375267, 82.3648682, -99.6352997, 97.3561249
3: -33.5641937, 70.9576569, -34.9264603, 73.7202835, -107.2844772, 105.8841095
4: -25.0775681, 70.1679382, -26.1118031, 72.8422928, -97.9198608, 96.2797394

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.60 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.98 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.98
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.98
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.98
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.98
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.98
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.98
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.98
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.98
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
Binary search (step 6): status=Status.VERIFIED, low=0.1240234, high=0.1250000, mid=0.1240234, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 7) starts
Candidate diff: 0.1245117


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.74 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.68 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.59 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -14.8509598, 83.8702927, -96.7212601, 84.5706635
1: -20.9004955, 82.9266129, -24.1805515, 99.6739273, -120.5744247, 107.1071625
2: -18.1388531, 82.9589615, -20.9075127, 99.5991058, -117.7379608, 103.8664551
3: -35.2660179, 74.2744598, -40.9821777, 88.5371628, -123.8031769, 115.2566376
4: -26.3749428, 73.4100494, -30.6739483, 87.2144394, -113.5893555, 104.0839996

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.65 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -14.8509598, 83.8702927, -99.0249863, 100.3664780
1: -24.6804409, 101.6211853, -24.1805515, 99.6739273, -124.3543701, 125.8017349
2: -21.4360867, 101.5054321, -20.9075127, 99.5991058, -121.0351868, 122.4129333
3: -41.7919083, 90.1225281, -40.9821777, 88.5371628, -130.3290405, 131.1046753
4: -31.3080330, 88.8759460, -30.6739483, 87.2144394, -118.5224609, 119.5498962

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.75 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.66 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.66
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -12.8509674, 69.7197113, -82.5706787, 82.5706787
1: -20.9004955, 82.9266129, -20.9004955, 82.9266129, -103.8271103, 103.8271103
2: -18.1388531, 82.9589615, -18.1388531, 82.9589615, -101.0978165, 101.0978165
3: -35.2660179, 74.2744598, -35.2660179, 74.2744598, -109.5404816, 109.5404816
4: -26.3749428, 73.4100494, -26.3749428, 73.4100494, -99.7849884, 99.7849884

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.71 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -15.1547003, 85.5155258, -98.3664932, 84.8744049
1: -20.9004955, 82.9266129, -24.6804409, 101.6211853, -122.5216827, 107.6070557
2: -18.1388531, 82.9589615, -21.4360867, 101.5054321, -119.6442871, 104.3950500
3: -35.2660179, 74.2744598, -41.7919083, 90.1225281, -125.3885498, 116.0663605
4: -26.3749428, 73.4100494, -31.3080330, 88.8759460, -115.2508850, 104.7180786

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.78 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -12.8509674, 69.7197113, -84.8744049, 98.3664932
1: -24.6804409, 101.6211853, -20.9004955, 82.9266129, -107.6070557, 122.5216827
2: -21.4360867, 101.5054321, -18.1388531, 82.9589615, -104.3950500, 119.6442871
3: -41.7919083, 90.1225281, -35.2660179, 74.2744598, -116.0663605, 125.3885498
4: -31.3080330, 88.8759460, -26.3749428, 73.4100494, -104.7180786, 115.2508850

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.82 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -15.1547003, 85.5155258, -100.6702194, 100.6702271
1: -24.6804409, 101.6211853, -24.6804409, 101.6211853, -126.3016281, 126.3016281
2: -21.4360867, 101.5054321, -21.4360867, 101.5054321, -122.9415207, 122.9415207
3: -41.7919083, 90.1225281, -41.7919083, 90.1225281, -131.9143982, 131.9143982
4: -31.3080330, 88.8759460, -31.3080330, 88.8759460, -120.1839752, 120.1839752

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.34 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.34
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.34
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.34
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.34
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.34
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.8509674, 69.7197113, -80.1642075, 68.8153381
1: -16.9038010, 66.4907379, -20.9004955, 82.9266129, -99.8304062, 87.3912354
2: -14.7949934, 66.5433578, -18.1388531, 82.9589615, -97.7539520, 84.6822128
3: -28.4008694, 59.6408424, -35.2660179, 74.2744598, -102.6753311, 94.9068604
4: -21.3142948, 59.0830536, -26.3749428, 73.4100494, -94.7243423, 85.4579849

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.8509674, 69.7197113, -82.4571075, 82.0802765
1: -20.6976452, 82.3365021, -20.9004955, 82.9266129, -103.6242523, 103.2369919
2: -17.9375267, 82.3648682, -18.1388531, 82.9589615, -100.8964767, 100.5037231
3: -34.9264603, 73.7202835, -35.2660179, 74.2744598, -109.2009201, 108.9862976
4: -26.1118031, 72.8422928, -26.3749428, 73.4100494, -99.5218430, 99.2172318

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -15.1547003, 85.5155258, -95.9600296, 71.1190643
1: -16.9038010, 66.4907379, -24.6804409, 101.6211853, -118.5249786, 91.1711807
2: -14.7949934, 66.5433578, -21.4360867, 101.5054321, -116.3004227, 87.9794464
3: -28.4008694, 59.6408424, -41.7919083, 90.1225281, -118.5233994, 101.4327393
4: -21.3142948, 59.0830536, -31.3080330, 88.8759460, -110.1902390, 90.3910828

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5405027, upper bound: 112.4936470
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -15.1547003, 85.5155258, -98.2529221, 84.3840027
1: -20.6976452, 82.3365021, -24.6804409, 101.6211853, -122.3188324, 107.0169449
2: -17.9375267, 82.3648682, -21.4360867, 101.5054321, -119.4429474, 103.8009567
3: -34.9264603, 73.7202835, -41.7919083, 90.1225281, -125.0489883, 115.5121918
4: -26.1118031, 72.8422928, -31.3080330, 88.8759460, -114.9877472, 104.1503296

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295
time: 0.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.53 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.5405027, upper bound: 112.4936470
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -10.4445028, 55.9643669, -66.4088669, 66.4088593
1: -16.9038010, 66.4907379, -16.9038010, 66.4907379, -83.3945312, 83.3945312
2: -14.7949934, 66.5433578, -14.7949934, 66.5433578, -81.3383484, 81.3383484
3: -28.4008694, 59.6408424, -28.4008694, 59.6408424, -88.0417099, 88.0417099
4: -21.3142948, 59.0830536, -21.3142948, 59.0830536, -80.3973465, 80.3973465

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.7374048, 69.2293091, -79.6738129, 68.7017746
1: -16.9038010, 66.4907379, -20.6976452, 82.3365021, -99.2402802, 87.1883774
2: -14.7949934, 66.5433578, -17.9375267, 82.3648682, -97.1598587, 84.4808655
3: -28.4008694, 59.6408424, -34.9264603, 73.7202835, -102.1211548, 94.5673065
4: -21.3142948, 59.0830536, -26.1118031, 72.8422928, -94.1565857, 85.1948547

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -10.4445028, 55.9643669, -68.7017670, 79.6738052
1: -20.6976452, 82.3365021, -16.9038010, 66.4907379, -87.1883774, 99.2402878
2: -17.9375267, 82.3648682, -14.7949934, 66.5433578, -84.4808731, 97.1598587
3: -34.9264603, 73.7202835, -28.4008694, 59.6408424, -94.5672989, 102.1211548
4: -26.1118031, 72.8422928, -21.3142948, 59.0830536, -85.1948547, 94.1565857

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.7374048, 69.2293091, -81.9667130, 81.9667130
1: -20.6976452, 82.3365021, -20.6976452, 82.3365021, -103.0341263, 103.0341339
2: -17.9375267, 82.3648682, -17.9375267, 82.3648682, -100.3023834, 100.3023834
3: -34.9264603, 73.7202835, -34.9264603, 73.7202835, -108.6467438, 108.6467438
4: -26.1118031, 72.8422928, -26.1118031, 72.8422928, -98.9540863, 98.9540939

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -14.6269550, 82.7033157, -93.1478195, 70.5913239
1: -16.9038010, 66.4907379, -23.8081474, 98.2509613, -115.1547546, 90.2988892
2: -14.7949934, 66.5433578, -20.6988335, 98.1297531, -112.9247437, 87.2421646
3: -28.4008694, 59.6408424, -40.2802467, 87.0169601, -115.4178314, 99.9210739
4: -21.3142948, 59.0830536, -30.1702652, 85.8516464, -107.1659393, 89.2533188

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4962782, upper bound: 112.4732156
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.63 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.56 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.56
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.56
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.56
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.56
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.56
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.56
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.56
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -10.4445028, 55.9643669, -66.0205383, 64.4386826
1: -16.2604694, 64.1127777, -16.9038010, 66.4907379, -82.7512054, 81.0165634
2: -14.2520180, 64.1702576, -14.7949934, 66.5433578, -80.7953568, 78.9652481
3: -27.2941570, 57.4110603, -28.4008694, 59.6408424, -86.9349976, 85.8119278
4: -20.4760780, 56.9184914, -21.3142948, 59.0830536, -79.5591202, 78.2327881

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -12.7374048, 69.2293091, -79.2854767, 66.7315750
1: -16.2604694, 64.1127777, -20.6976452, 82.3365021, -98.5969620, 84.8104095
2: -14.2520180, 64.1702576, -17.9375267, 82.3648682, -96.6168671, 82.1077805
3: -27.2941570, 57.4110603, -34.9264603, 73.7202835, -101.0144424, 92.3375092
4: -20.4760780, 56.9184914, -26.1118031, 72.8422928, -93.3183670, 83.0302811

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -10.4445028, 55.9643669, -68.2222595, 77.2292023
1: -19.9063568, 79.3842850, -16.9038010, 66.4907379, -86.3970947, 96.2880783
2: -17.2704391, 79.4186096, -14.7949934, 66.5433578, -83.8137894, 94.2136002
3: -33.5641937, 70.9576569, -28.4008694, 59.6408424, -93.2050323, 99.3585281
4: -25.0775681, 70.1679382, -21.3142948, 59.0830536, -84.1606216, 91.4822311

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -12.7374048, 69.2293091, -81.4871902, 79.5221100
1: -19.9063568, 79.3842850, -20.6976452, 82.3365021, -102.2428513, 100.0819244
2: -17.2704391, 79.4186096, -17.9375267, 82.3648682, -99.6352997, 97.3561249
3: -33.5641937, 70.9576569, -34.9264603, 73.7202835, -107.2844772, 105.8841095
4: -25.0775681, 70.1679382, -26.1118031, 72.8422928, -97.9198608, 96.2797394

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.23 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.23
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.23
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.23
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.23
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.23
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.23
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.23
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.23
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
Binary search (step 7): status=Status.VERIFIED, low=0.1245117, high=0.1250000, mid=0.1245117, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 8) starts
Candidate diff: 0.1247559


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.78 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.65 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.65
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -14.8509598, 83.8702927, -96.7212601, 84.5706635
1: -20.9004955, 82.9266129, -24.1805515, 99.6739273, -120.5744247, 107.1071625
2: -18.1388531, 82.9589615, -20.9075127, 99.5991058, -117.7379608, 103.8664551
3: -35.2660179, 74.2744598, -40.9821777, 88.5371628, -123.8031769, 115.2566376
4: -26.3749428, 73.4100494, -30.6739483, 87.2144394, -113.5893555, 104.0839996

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.66 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -14.8509598, 83.8702927, -99.0249863, 100.3664780
1: -24.6804409, 101.6211853, -24.1805515, 99.6739273, -124.3543701, 125.8017349
2: -21.4360867, 101.5054321, -20.9075127, 99.5991058, -121.0351868, 122.4129333
3: -41.7919083, 90.1225281, -40.9821777, 88.5371628, -130.3290405, 131.1046753
4: -31.3080330, 88.8759460, -30.6739483, 87.2144394, -118.5224609, 119.5498962

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.72 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.58 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -12.8509674, 69.7197113, -82.5706787, 82.5706787
1: -20.9004955, 82.9266129, -20.9004955, 82.9266129, -103.8271103, 103.8271103
2: -18.1388531, 82.9589615, -18.1388531, 82.9589615, -101.0978165, 101.0978165
3: -35.2660179, 74.2744598, -35.2660179, 74.2744598, -109.5404816, 109.5404816
4: -26.3749428, 73.4100494, -26.3749428, 73.4100494, -99.7849884, 99.7849884

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.80 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -15.1547003, 85.5155258, -98.3664932, 84.8744049
1: -20.9004955, 82.9266129, -24.6804409, 101.6211853, -122.5216827, 107.6070557
2: -18.1388531, 82.9589615, -21.4360867, 101.5054321, -119.6442871, 104.3950500
3: -35.2660179, 74.2744598, -41.7919083, 90.1225281, -125.3885498, 116.0663605
4: -26.3749428, 73.4100494, -31.3080330, 88.8759460, -115.2508850, 104.7180786

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.79 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -12.8509674, 69.7197113, -84.8744049, 98.3664932
1: -24.6804409, 101.6211853, -20.9004955, 82.9266129, -107.6070557, 122.5216827
2: -21.4360867, 101.5054321, -18.1388531, 82.9589615, -104.3950500, 119.6442871
3: -41.7919083, 90.1225281, -35.2660179, 74.2744598, -116.0663605, 125.3885498
4: -31.3080330, 88.8759460, -26.3749428, 73.4100494, -104.7180786, 115.2508850

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.76 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -15.1547003, 85.5155258, -100.6702194, 100.6702271
1: -24.6804409, 101.6211853, -24.6804409, 101.6211853, -126.3016281, 126.3016281
2: -21.4360867, 101.5054321, -21.4360867, 101.5054321, -122.9415207, 122.9415207
3: -41.7919083, 90.1225281, -41.7919083, 90.1225281, -131.9143982, 131.9143982
4: -31.3080330, 88.8759460, -31.3080330, 88.8759460, -120.1839752, 120.1839752

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.38 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.38
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.38
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.38
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.38
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.8509674, 69.7197113, -80.1642075, 68.8153381
1: -16.9038010, 66.4907379, -20.9004955, 82.9266129, -99.8304062, 87.3912354
2: -14.7949934, 66.5433578, -18.1388531, 82.9589615, -97.7539520, 84.6822128
3: -28.4008694, 59.6408424, -35.2660179, 74.2744598, -102.6753311, 94.9068604
4: -21.3142948, 59.0830536, -26.3749428, 73.4100494, -94.7243423, 85.4579849

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.8509674, 69.7197113, -82.4571075, 82.0802765
1: -20.6976452, 82.3365021, -20.9004955, 82.9266129, -103.6242523, 103.2369919
2: -17.9375267, 82.3648682, -18.1388531, 82.9589615, -100.8964767, 100.5037231
3: -34.9264603, 73.7202835, -35.2660179, 74.2744598, -109.2009201, 108.9862976
4: -26.1118031, 72.8422928, -26.3749428, 73.4100494, -99.5218430, 99.2172318

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -15.1547003, 85.5155258, -95.9600296, 71.1190643
1: -16.9038010, 66.4907379, -24.6804409, 101.6211853, -118.5249786, 91.1711807
2: -14.7949934, 66.5433578, -21.4360867, 101.5054321, -116.3004227, 87.9794464
3: -28.4008694, 59.6408424, -41.7919083, 90.1225281, -118.5233994, 101.4327393
4: -21.3142948, 59.0830536, -31.3080330, 88.8759460, -110.1902390, 90.3910828

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5405036, upper bound: 112.4936470
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -15.1547003, 85.5155258, -98.2529221, 84.3840027
1: -20.6976452, 82.3365021, -24.6804409, 101.6211853, -122.3188324, 107.0169449
2: -17.9375267, 82.3648682, -21.4360867, 101.5054321, -119.4429474, 103.8009567
3: -34.9264603, 73.7202835, -41.7919083, 90.1225281, -125.0489883, 115.5121918
4: -26.1118031, 72.8422928, -31.3080330, 88.8759460, -114.9877472, 104.1503296

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295
time: 0.86 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.81 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.81
Output dim: 3, lower bound: -112.5405036, upper bound: 112.4936470
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.81
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.81
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.81
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -10.4445028, 55.9643669, -66.4088669, 66.4088593
1: -16.9038010, 66.4907379, -16.9038010, 66.4907379, -83.3945312, 83.3945312
2: -14.7949934, 66.5433578, -14.7949934, 66.5433578, -81.3383484, 81.3383484
3: -28.4008694, 59.6408424, -28.4008694, 59.6408424, -88.0417099, 88.0417099
4: -21.3142948, 59.0830536, -21.3142948, 59.0830536, -80.3973465, 80.3973465

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.7374048, 69.2293091, -79.6738129, 68.7017746
1: -16.9038010, 66.4907379, -20.6976452, 82.3365021, -99.2402802, 87.1883774
2: -14.7949934, 66.5433578, -17.9375267, 82.3648682, -97.1598587, 84.4808655
3: -28.4008694, 59.6408424, -34.9264603, 73.7202835, -102.1211548, 94.5673065
4: -21.3142948, 59.0830536, -26.1118031, 72.8422928, -94.1565857, 85.1948547

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -10.4445028, 55.9643669, -68.7017670, 79.6738052
1: -20.6976452, 82.3365021, -16.9038010, 66.4907379, -87.1883774, 99.2402878
2: -17.9375267, 82.3648682, -14.7949934, 66.5433578, -84.4808731, 97.1598587
3: -34.9264603, 73.7202835, -28.4008694, 59.6408424, -94.5672989, 102.1211548
4: -26.1118031, 72.8422928, -21.3142948, 59.0830536, -85.1948547, 94.1565857

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.7374048, 69.2293091, -81.9667130, 81.9667130
1: -20.6976452, 82.3365021, -20.6976452, 82.3365021, -103.0341263, 103.0341339
2: -17.9375267, 82.3648682, -17.9375267, 82.3648682, -100.3023834, 100.3023834
3: -34.9264603, 73.7202835, -34.9264603, 73.7202835, -108.6467438, 108.6467438
4: -26.1118031, 72.8422928, -26.1118031, 72.8422928, -98.9540863, 98.9540939

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -14.6269550, 82.7033157, -93.1478195, 70.5913239
1: -16.9038010, 66.4907379, -23.8081474, 98.2509613, -115.1547546, 90.2988892
2: -14.7949934, 66.5433578, -20.6988335, 98.1297531, -112.9247437, 87.2421646
3: -28.4008694, 59.6408424, -40.2802467, 87.0169601, -115.4178314, 99.9210739
4: -21.3142948, 59.0830536, -30.1702652, 85.8516464, -107.1659393, 89.2533188

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4962782, upper bound: 112.4732156
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.73 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -10.4445028, 55.9643669, -66.0205383, 64.4386826
1: -16.2604694, 64.1127777, -16.9038010, 66.4907379, -82.7512054, 81.0165634
2: -14.2520180, 64.1702576, -14.7949934, 66.5433578, -80.7953568, 78.9652481
3: -27.2941570, 57.4110603, -28.4008694, 59.6408424, -86.9349976, 85.8119278
4: -20.4760780, 56.9184914, -21.3142948, 59.0830536, -79.5591202, 78.2327881

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -12.7374048, 69.2293091, -79.2854767, 66.7315750
1: -16.2604694, 64.1127777, -20.6976452, 82.3365021, -98.5969620, 84.8104095
2: -14.2520180, 64.1702576, -17.9375267, 82.3648682, -96.6168671, 82.1077805
3: -27.2941570, 57.4110603, -34.9264603, 73.7202835, -101.0144424, 92.3375092
4: -20.4760780, 56.9184914, -26.1118031, 72.8422928, -93.3183670, 83.0302811

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -10.4445028, 55.9643669, -68.2222595, 77.2292023
1: -19.9063568, 79.3842850, -16.9038010, 66.4907379, -86.3970947, 96.2880783
2: -17.2704391, 79.4186096, -14.7949934, 66.5433578, -83.8137894, 94.2136002
3: -33.5641937, 70.9576569, -28.4008694, 59.6408424, -93.2050323, 99.3585281
4: -25.0775681, 70.1679382, -21.3142948, 59.0830536, -84.1606216, 91.4822311

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -12.7374048, 69.2293091, -81.4871902, 79.5221100
1: -19.9063568, 79.3842850, -20.6976452, 82.3365021, -102.2428513, 100.0819244
2: -17.2704391, 79.4186096, -17.9375267, 82.3648682, -99.6352997, 97.3561249
3: -33.5641937, 70.9576569, -34.9264603, 73.7202835, -107.2844772, 105.8841095
4: -25.0775681, 70.1679382, -26.1118031, 72.8422928, -97.9198608, 96.2797394

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.67 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.37 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.37
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.37
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.37
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.37
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.37
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.37
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.37
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.37
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
Binary search (step 8): status=Status.VERIFIED, low=0.1247559, high=0.1250000, mid=0.1247559, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 9) starts
Candidate diff: 0.1248779


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.76 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.64 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.58 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -14.8509598, 83.8702927, -96.7212601, 84.5706635
1: -20.9004955, 82.9266129, -24.1805515, 99.6739273, -120.5744247, 107.1071625
2: -18.1388531, 82.9589615, -20.9075127, 99.5991058, -117.7379608, 103.8664551
3: -35.2660179, 74.2744598, -40.9821777, 88.5371628, -123.8031769, 115.2566376
4: -26.3749428, 73.4100494, -30.6739483, 87.2144394, -113.5893555, 104.0839996

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.58 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.70 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -14.8509598, 83.8702927, -99.0249863, 100.3664780
1: -24.6804409, 101.6211853, -24.1805515, 99.6739273, -124.3543701, 125.8017349
2: -21.4360867, 101.5054321, -20.9075127, 99.5991058, -121.0351868, 122.4129333
3: -41.7919083, 90.1225281, -40.9821777, 88.5371628, -130.3290405, 131.1046753
4: -31.3080330, 88.8759460, -30.6739483, 87.2144394, -118.5224609, 119.5498962

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.71 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.73 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.73
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -12.8509674, 69.7197113, -82.5706787, 82.5706787
1: -20.9004955, 82.9266129, -20.9004955, 82.9266129, -103.8271103, 103.8271103
2: -18.1388531, 82.9589615, -18.1388531, 82.9589615, -101.0978165, 101.0978165
3: -35.2660179, 74.2744598, -35.2660179, 74.2744598, -109.5404816, 109.5404816
4: -26.3749428, 73.4100494, -26.3749428, 73.4100494, -99.7849884, 99.7849884

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.57 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -15.1547003, 85.5155258, -98.3664932, 84.8744049
1: -20.9004955, 82.9266129, -24.6804409, 101.6211853, -122.5216827, 107.6070557
2: -18.1388531, 82.9589615, -21.4360867, 101.5054321, -119.6442871, 104.3950500
3: -35.2660179, 74.2744598, -41.7919083, 90.1225281, -125.3885498, 116.0663605
4: -26.3749428, 73.4100494, -31.3080330, 88.8759460, -115.2508850, 104.7180786

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.94 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -12.8509674, 69.7197113, -84.8744049, 98.3664932
1: -24.6804409, 101.6211853, -20.9004955, 82.9266129, -107.6070557, 122.5216827
2: -21.4360867, 101.5054321, -18.1388531, 82.9589615, -104.3950500, 119.6442871
3: -41.7919083, 90.1225281, -35.2660179, 74.2744598, -116.0663605, 125.3885498
4: -31.3080330, 88.8759460, -26.3749428, 73.4100494, -104.7180786, 115.2508850

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.75 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -15.1547003, 85.5155258, -100.6702194, 100.6702271
1: -24.6804409, 101.6211853, -24.6804409, 101.6211853, -126.3016281, 126.3016281
2: -21.4360867, 101.5054321, -21.4360867, 101.5054321, -122.9415207, 122.9415207
3: -41.7919083, 90.1225281, -41.7919083, 90.1225281, -131.9143982, 131.9143982
4: -31.3080330, 88.8759460, -31.3080330, 88.8759460, -120.1839752, 120.1839752

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.46 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.8509674, 69.7197113, -80.1642075, 68.8153381
1: -16.9038010, 66.4907379, -20.9004955, 82.9266129, -99.8304062, 87.3912354
2: -14.7949934, 66.5433578, -18.1388531, 82.9589615, -97.7539520, 84.6822128
3: -28.4008694, 59.6408424, -35.2660179, 74.2744598, -102.6753311, 94.9068604
4: -21.3142948, 59.0830536, -26.3749428, 73.4100494, -94.7243423, 85.4579849

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.8509674, 69.7197113, -82.4571075, 82.0802765
1: -20.6976452, 82.3365021, -20.9004955, 82.9266129, -103.6242523, 103.2369919
2: -17.9375267, 82.3648682, -18.1388531, 82.9589615, -100.8964767, 100.5037231
3: -34.9264603, 73.7202835, -35.2660179, 74.2744598, -109.2009201, 108.9862976
4: -26.1118031, 72.8422928, -26.3749428, 73.4100494, -99.5218430, 99.2172318

Time for backsubstitution: 2.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -15.1547003, 85.5155258, -95.9600296, 71.1190643
1: -16.9038010, 66.4907379, -24.6804409, 101.6211853, -118.5249786, 91.1711807
2: -14.7949934, 66.5433578, -21.4360867, 101.5054321, -116.3004227, 87.9794464
3: -28.4008694, 59.6408424, -41.7919083, 90.1225281, -118.5233994, 101.4327393
4: -21.3142948, 59.0830536, -31.3080330, 88.8759460, -110.1902390, 90.3910828

Time for backsubstitution: 2.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5405040, upper bound: 112.4936470
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -15.1547003, 85.5155258, -98.2529221, 84.3840027
1: -20.6976452, 82.3365021, -24.6804409, 101.6211853, -122.3188324, 107.0169449
2: -17.9375267, 82.3648682, -21.4360867, 101.5054321, -119.4429474, 103.8009567
3: -34.9264603, 73.7202835, -41.7919083, 90.1225281, -125.0489883, 115.5121918
4: -26.1118031, 72.8422928, -31.3080330, 88.8759460, -114.9877472, 104.1503296

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295
time: 0.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.80 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.5405040, upper bound: 112.4936470
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.80
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -10.4445028, 55.9643669, -66.4088669, 66.4088593
1: -16.9038010, 66.4907379, -16.9038010, 66.4907379, -83.3945312, 83.3945312
2: -14.7949934, 66.5433578, -14.7949934, 66.5433578, -81.3383484, 81.3383484
3: -28.4008694, 59.6408424, -28.4008694, 59.6408424, -88.0417099, 88.0417099
4: -21.3142948, 59.0830536, -21.3142948, 59.0830536, -80.3973465, 80.3973465

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.7374048, 69.2293091, -79.6738129, 68.7017746
1: -16.9038010, 66.4907379, -20.6976452, 82.3365021, -99.2402802, 87.1883774
2: -14.7949934, 66.5433578, -17.9375267, 82.3648682, -97.1598587, 84.4808655
3: -28.4008694, 59.6408424, -34.9264603, 73.7202835, -102.1211548, 94.5673065
4: -21.3142948, 59.0830536, -26.1118031, 72.8422928, -94.1565857, 85.1948547

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -10.4445028, 55.9643669, -68.7017670, 79.6738052
1: -20.6976452, 82.3365021, -16.9038010, 66.4907379, -87.1883774, 99.2402878
2: -17.9375267, 82.3648682, -14.7949934, 66.5433578, -84.4808731, 97.1598587
3: -34.9264603, 73.7202835, -28.4008694, 59.6408424, -94.5672989, 102.1211548
4: -26.1118031, 72.8422928, -21.3142948, 59.0830536, -85.1948547, 94.1565857

Time for backsubstitution: 2.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.7374048, 69.2293091, -81.9667130, 81.9667130
1: -20.6976452, 82.3365021, -20.6976452, 82.3365021, -103.0341263, 103.0341339
2: -17.9375267, 82.3648682, -17.9375267, 82.3648682, -100.3023834, 100.3023834
3: -34.9264603, 73.7202835, -34.9264603, 73.7202835, -108.6467438, 108.6467438
4: -26.1118031, 72.8422928, -26.1118031, 72.8422928, -98.9540863, 98.9540939

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -14.6269550, 82.7033157, -93.1478195, 70.5913239
1: -16.9038010, 66.4907379, -23.8081474, 98.2509613, -115.1547546, 90.2988892
2: -14.7949934, 66.5433578, -20.6988335, 98.1297531, -112.9247437, 87.2421646
3: -28.4008694, 59.6408424, -40.2802467, 87.0169601, -115.4178314, 99.9210739
4: -21.3142948, 59.0830536, -30.1702652, 85.8516464, -107.1659393, 89.2533188

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4962782, upper bound: 112.4732156
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.79 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.79
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.79
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.79
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.79
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.79
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.79
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.79
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -10.4445028, 55.9643669, -66.0205383, 64.4386826
1: -16.2604694, 64.1127777, -16.9038010, 66.4907379, -82.7512054, 81.0165634
2: -14.2520180, 64.1702576, -14.7949934, 66.5433578, -80.7953568, 78.9652481
3: -27.2941570, 57.4110603, -28.4008694, 59.6408424, -86.9349976, 85.8119278
4: -20.4760780, 56.9184914, -21.3142948, 59.0830536, -79.5591202, 78.2327881

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -12.7374048, 69.2293091, -79.2854767, 66.7315750
1: -16.2604694, 64.1127777, -20.6976452, 82.3365021, -98.5969620, 84.8104095
2: -14.2520180, 64.1702576, -17.9375267, 82.3648682, -96.6168671, 82.1077805
3: -27.2941570, 57.4110603, -34.9264603, 73.7202835, -101.0144424, 92.3375092
4: -20.4760780, 56.9184914, -26.1118031, 72.8422928, -93.3183670, 83.0302811

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -10.4445028, 55.9643669, -68.2222595, 77.2292023
1: -19.9063568, 79.3842850, -16.9038010, 66.4907379, -86.3970947, 96.2880783
2: -17.2704391, 79.4186096, -14.7949934, 66.5433578, -83.8137894, 94.2136002
3: -33.5641937, 70.9576569, -28.4008694, 59.6408424, -93.2050323, 99.3585281
4: -25.0775681, 70.1679382, -21.3142948, 59.0830536, -84.1606216, 91.4822311

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -12.7374048, 69.2293091, -81.4871902, 79.5221100
1: -19.9063568, 79.3842850, -20.6976452, 82.3365021, -102.2428513, 100.0819244
2: -17.2704391, 79.4186096, -17.9375267, 82.3648682, -99.6352997, 97.3561249
3: -33.5641937, 70.9576569, -34.9264603, 73.7202835, -107.2844772, 105.8841095
4: -25.0775681, 70.1679382, -26.1118031, 72.8422928, -97.9198608, 96.2797394

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.64 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.07 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
Binary search (step 9): status=Status.VERIFIED, low=0.1248779, high=0.1250000, mid=0.1248779, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 10) starts
Candidate diff: 0.1249390


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.91 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.80 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.80
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.80
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -14.8509598, 83.8702927, -96.7212601, 84.5706635
1: -20.9004955, 82.9266129, -24.1805515, 99.6739273, -120.5744247, 107.1071625
2: -18.1388531, 82.9589615, -20.9075127, 99.5991058, -117.7379608, 103.8664551
3: -35.2660179, 74.2744598, -40.9821777, 88.5371628, -123.8031769, 115.2566376
4: -26.3749428, 73.4100494, -30.6739483, 87.2144394, -113.5893555, 104.0839996

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.78 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -14.8509598, 83.8702927, -99.0249863, 100.3664780
1: -24.6804409, 101.6211853, -24.1805515, 99.6739273, -124.3543701, 125.8017349
2: -21.4360867, 101.5054321, -20.9075127, 99.5991058, -121.0351868, 122.4129333
3: -41.7919083, 90.1225281, -40.9821777, 88.5371628, -130.3290405, 131.1046753
4: -31.3080330, 88.8759460, -30.6739483, 87.2144394, -118.5224609, 119.5498962

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.68 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.53 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.53
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.53
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.53
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.53
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -12.8509674, 69.7197113, -82.5706787, 82.5706787
1: -20.9004955, 82.9266129, -20.9004955, 82.9266129, -103.8271103, 103.8271103
2: -18.1388531, 82.9589615, -18.1388531, 82.9589615, -101.0978165, 101.0978165
3: -35.2660179, 74.2744598, -35.2660179, 74.2744598, -109.5404816, 109.5404816
4: -26.3749428, 73.4100494, -26.3749428, 73.4100494, -99.7849884, 99.7849884

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.80 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -15.1547003, 85.5155258, -98.3664932, 84.8744049
1: -20.9004955, 82.9266129, -24.6804409, 101.6211853, -122.5216827, 107.6070557
2: -18.1388531, 82.9589615, -21.4360867, 101.5054321, -119.6442871, 104.3950500
3: -35.2660179, 74.2744598, -41.7919083, 90.1225281, -125.3885498, 116.0663605
4: -26.3749428, 73.4100494, -31.3080330, 88.8759460, -115.2508850, 104.7180786

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.86 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -12.8509674, 69.7197113, -84.8744049, 98.3664932
1: -24.6804409, 101.6211853, -20.9004955, 82.9266129, -107.6070557, 122.5216827
2: -21.4360867, 101.5054321, -18.1388531, 82.9589615, -104.3950500, 119.6442871
3: -41.7919083, 90.1225281, -35.2660179, 74.2744598, -116.0663605, 125.3885498
4: -31.3080330, 88.8759460, -26.3749428, 73.4100494, -104.7180786, 115.2508850

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.77 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -15.1547003, 85.5155258, -100.6702194, 100.6702271
1: -24.6804409, 101.6211853, -24.6804409, 101.6211853, -126.3016281, 126.3016281
2: -21.4360867, 101.5054321, -21.4360867, 101.5054321, -122.9415207, 122.9415207
3: -41.7919083, 90.1225281, -41.7919083, 90.1225281, -131.9143982, 131.9143982
4: -31.3080330, 88.8759460, -31.3080330, 88.8759460, -120.1839752, 120.1839752

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.44 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.44
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.44
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.44
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.44
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.44
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.44
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.44
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.44
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.8509674, 69.7197113, -80.1642075, 68.8153381
1: -16.9038010, 66.4907379, -20.9004955, 82.9266129, -99.8304062, 87.3912354
2: -14.7949934, 66.5433578, -18.1388531, 82.9589615, -97.7539520, 84.6822128
3: -28.4008694, 59.6408424, -35.2660179, 74.2744598, -102.6753311, 94.9068604
4: -21.3142948, 59.0830536, -26.3749428, 73.4100494, -94.7243423, 85.4579849

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.8509674, 69.7197113, -82.4571075, 82.0802765
1: -20.6976452, 82.3365021, -20.9004955, 82.9266129, -103.6242523, 103.2369919
2: -17.9375267, 82.3648682, -18.1388531, 82.9589615, -100.8964767, 100.5037231
3: -34.9264603, 73.7202835, -35.2660179, 74.2744598, -109.2009201, 108.9862976
4: -26.1118031, 72.8422928, -26.3749428, 73.4100494, -99.5218430, 99.2172318

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -15.1547003, 85.5155258, -95.9600296, 71.1190643
1: -16.9038010, 66.4907379, -24.6804409, 101.6211853, -118.5249786, 91.1711807
2: -14.7949934, 66.5433578, -21.4360867, 101.5054321, -116.3004227, 87.9794464
3: -28.4008694, 59.6408424, -41.7919083, 90.1225281, -118.5233994, 101.4327393
4: -21.3142948, 59.0830536, -31.3080330, 88.8759460, -110.1902390, 90.3910828

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5405043, upper bound: 112.4936470
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -15.1547003, 85.5155258, -98.2529221, 84.3840027
1: -20.6976452, 82.3365021, -24.6804409, 101.6211853, -122.3188324, 107.0169449
2: -17.9375267, 82.3648682, -21.4360867, 101.5054321, -119.4429474, 103.8009567
3: -34.9264603, 73.7202835, -41.7919083, 90.1225281, -125.0489883, 115.5121918
4: -26.1118031, 72.8422928, -31.3080330, 88.8759460, -114.9877472, 104.1503296

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295
time: 0.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.53 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.5405043, upper bound: 112.4936470
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.53
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -10.4445028, 55.9643669, -66.4088669, 66.4088593
1: -16.9038010, 66.4907379, -16.9038010, 66.4907379, -83.3945312, 83.3945312
2: -14.7949934, 66.5433578, -14.7949934, 66.5433578, -81.3383484, 81.3383484
3: -28.4008694, 59.6408424, -28.4008694, 59.6408424, -88.0417099, 88.0417099
4: -21.3142948, 59.0830536, -21.3142948, 59.0830536, -80.3973465, 80.3973465

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.7374048, 69.2293091, -79.6738129, 68.7017746
1: -16.9038010, 66.4907379, -20.6976452, 82.3365021, -99.2402802, 87.1883774
2: -14.7949934, 66.5433578, -17.9375267, 82.3648682, -97.1598587, 84.4808655
3: -28.4008694, 59.6408424, -34.9264603, 73.7202835, -102.1211548, 94.5673065
4: -21.3142948, 59.0830536, -26.1118031, 72.8422928, -94.1565857, 85.1948547

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -10.4445028, 55.9643669, -68.7017670, 79.6738052
1: -20.6976452, 82.3365021, -16.9038010, 66.4907379, -87.1883774, 99.2402878
2: -17.9375267, 82.3648682, -14.7949934, 66.5433578, -84.4808731, 97.1598587
3: -34.9264603, 73.7202835, -28.4008694, 59.6408424, -94.5672989, 102.1211548
4: -26.1118031, 72.8422928, -21.3142948, 59.0830536, -85.1948547, 94.1565857

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.7374048, 69.2293091, -81.9667130, 81.9667130
1: -20.6976452, 82.3365021, -20.6976452, 82.3365021, -103.0341263, 103.0341339
2: -17.9375267, 82.3648682, -17.9375267, 82.3648682, -100.3023834, 100.3023834
3: -34.9264603, 73.7202835, -34.9264603, 73.7202835, -108.6467438, 108.6467438
4: -26.1118031, 72.8422928, -26.1118031, 72.8422928, -98.9540863, 98.9540939

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -14.6269550, 82.7033157, -93.1478195, 70.5913239
1: -16.9038010, 66.4907379, -23.8081474, 98.2509613, -115.1547546, 90.2988892
2: -14.7949934, 66.5433578, -20.6988335, 98.1297531, -112.9247437, 87.2421646
3: -28.4008694, 59.6408424, -40.2802467, 87.0169601, -115.4178314, 99.9210739
4: -21.3142948, 59.0830536, -30.1702652, 85.8516464, -107.1659393, 89.2533188

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4962782, upper bound: 112.4732156
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.61 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.66 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.66
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.66
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.66
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.66
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.66
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.66
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.66
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -10.4445028, 55.9643669, -66.0205383, 64.4386826
1: -16.2604694, 64.1127777, -16.9038010, 66.4907379, -82.7512054, 81.0165634
2: -14.2520180, 64.1702576, -14.7949934, 66.5433578, -80.7953568, 78.9652481
3: -27.2941570, 57.4110603, -28.4008694, 59.6408424, -86.9349976, 85.8119278
4: -20.4760780, 56.9184914, -21.3142948, 59.0830536, -79.5591202, 78.2327881

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -12.7374048, 69.2293091, -79.2854767, 66.7315750
1: -16.2604694, 64.1127777, -20.6976452, 82.3365021, -98.5969620, 84.8104095
2: -14.2520180, 64.1702576, -17.9375267, 82.3648682, -96.6168671, 82.1077805
3: -27.2941570, 57.4110603, -34.9264603, 73.7202835, -101.0144424, 92.3375092
4: -20.4760780, 56.9184914, -26.1118031, 72.8422928, -93.3183670, 83.0302811

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -10.4445028, 55.9643669, -68.2222595, 77.2292023
1: -19.9063568, 79.3842850, -16.9038010, 66.4907379, -86.3970947, 96.2880783
2: -17.2704391, 79.4186096, -14.7949934, 66.5433578, -83.8137894, 94.2136002
3: -33.5641937, 70.9576569, -28.4008694, 59.6408424, -93.2050323, 99.3585281
4: -25.0775681, 70.1679382, -21.3142948, 59.0830536, -84.1606216, 91.4822311

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -12.7374048, 69.2293091, -81.4871902, 79.5221100
1: -19.9063568, 79.3842850, -20.6976452, 82.3365021, -102.2428513, 100.0819244
2: -17.2704391, 79.4186096, -17.9375267, 82.3648682, -99.6352997, 97.3561249
3: -33.5641937, 70.9576569, -34.9264603, 73.7202835, -107.2844772, 105.8841095
4: -25.0775681, 70.1679382, -26.1118031, 72.8422928, -97.9198608, 96.2797394

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.20 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.20
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.20
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.20
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.20
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.20
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.20
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.20
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.20
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
Binary search (step 10): status=Status.VERIFIED, low=0.1249390, high=0.1250000, mid=0.1249390, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 11) starts
Candidate diff: 0.1249695


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.74 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.59 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -14.8509598, 83.8702927, -96.7212601, 84.5706635
1: -20.9004955, 82.9266129, -24.1805515, 99.6739273, -120.5744247, 107.1071625
2: -18.1388531, 82.9589615, -20.9075127, 99.5991058, -117.7379608, 103.8664551
3: -35.2660179, 74.2744598, -40.9821777, 88.5371628, -123.8031769, 115.2566376
4: -26.3749428, 73.4100494, -30.6739483, 87.2144394, -113.5893555, 104.0839996

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.57 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.85 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -14.8509598, 83.8702927, -99.0249863, 100.3664780
1: -24.6804409, 101.6211853, -24.1805515, 99.6739273, -124.3543701, 125.8017349
2: -21.4360867, 101.5054321, -20.9075127, 99.5991058, -121.0351868, 122.4129333
3: -41.7919083, 90.1225281, -40.9821777, 88.5371628, -130.3290405, 131.1046753
4: -31.3080330, 88.8759460, -30.6739483, 87.2144394, -118.5224609, 119.5498962

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.70 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.65 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -12.8509674, 69.7197113, -82.5706787, 82.5706787
1: -20.9004955, 82.9266129, -20.9004955, 82.9266129, -103.8271103, 103.8271103
2: -18.1388531, 82.9589615, -18.1388531, 82.9589615, -101.0978165, 101.0978165
3: -35.2660179, 74.2744598, -35.2660179, 74.2744598, -109.5404816, 109.5404816
4: -26.3749428, 73.4100494, -26.3749428, 73.4100494, -99.7849884, 99.7849884

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.72 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -15.1547003, 85.5155258, -98.3664932, 84.8744049
1: -20.9004955, 82.9266129, -24.6804409, 101.6211853, -122.5216827, 107.6070557
2: -18.1388531, 82.9589615, -21.4360867, 101.5054321, -119.6442871, 104.3950500
3: -35.2660179, 74.2744598, -41.7919083, 90.1225281, -125.3885498, 116.0663605
4: -26.3749428, 73.4100494, -31.3080330, 88.8759460, -115.2508850, 104.7180786

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.72 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -12.8509674, 69.7197113, -84.8744049, 98.3664932
1: -24.6804409, 101.6211853, -20.9004955, 82.9266129, -107.6070557, 122.5216827
2: -21.4360867, 101.5054321, -18.1388531, 82.9589615, -104.3950500, 119.6442871
3: -41.7919083, 90.1225281, -35.2660179, 74.2744598, -116.0663605, 125.3885498
4: -31.3080330, 88.8759460, -26.3749428, 73.4100494, -104.7180786, 115.2508850

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.75 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -15.1547003, 85.5155258, -100.6702194, 100.6702271
1: -24.6804409, 101.6211853, -24.6804409, 101.6211853, -126.3016281, 126.3016281
2: -21.4360867, 101.5054321, -21.4360867, 101.5054321, -122.9415207, 122.9415207
3: -41.7919083, 90.1225281, -41.7919083, 90.1225281, -131.9143982, 131.9143982
4: -31.3080330, 88.8759460, -31.3080330, 88.8759460, -120.1839752, 120.1839752

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.52 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.52
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.52
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.52
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.52
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.8509674, 69.7197113, -80.1642075, 68.8153381
1: -16.9038010, 66.4907379, -20.9004955, 82.9266129, -99.8304062, 87.3912354
2: -14.7949934, 66.5433578, -18.1388531, 82.9589615, -97.7539520, 84.6822128
3: -28.4008694, 59.6408424, -35.2660179, 74.2744598, -102.6753311, 94.9068604
4: -21.3142948, 59.0830536, -26.3749428, 73.4100494, -94.7243423, 85.4579849

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.8509674, 69.7197113, -82.4571075, 82.0802765
1: -20.6976452, 82.3365021, -20.9004955, 82.9266129, -103.6242523, 103.2369919
2: -17.9375267, 82.3648682, -18.1388531, 82.9589615, -100.8964767, 100.5037231
3: -34.9264603, 73.7202835, -35.2660179, 74.2744598, -109.2009201, 108.9862976
4: -26.1118031, 72.8422928, -26.3749428, 73.4100494, -99.5218430, 99.2172318

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -15.1547003, 85.5155258, -95.9600296, 71.1190643
1: -16.9038010, 66.4907379, -24.6804409, 101.6211853, -118.5249786, 91.1711807
2: -14.7949934, 66.5433578, -21.4360867, 101.5054321, -116.3004227, 87.9794464
3: -28.4008694, 59.6408424, -41.7919083, 90.1225281, -118.5233994, 101.4327393
4: -21.3142948, 59.0830536, -31.3080330, 88.8759460, -110.1902390, 90.3910828

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5405044, upper bound: 112.4936470
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -15.1547003, 85.5155258, -98.2529221, 84.3840027
1: -20.6976452, 82.3365021, -24.6804409, 101.6211853, -122.3188324, 107.0169449
2: -17.9375267, 82.3648682, -21.4360867, 101.5054321, -119.4429474, 103.8009567
3: -34.9264603, 73.7202835, -41.7919083, 90.1225281, -125.0489883, 115.5121918
4: -26.1118031, 72.8422928, -31.3080330, 88.8759460, -114.9877472, 104.1503296

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295
time: 0.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.62 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.62
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.62
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.62
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.62
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.62
Output dim: 3, lower bound: -112.5405044, upper bound: 112.4936470
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.62
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -10.4445028, 55.9643669, -66.4088669, 66.4088593
1: -16.9038010, 66.4907379, -16.9038010, 66.4907379, -83.3945312, 83.3945312
2: -14.7949934, 66.5433578, -14.7949934, 66.5433578, -81.3383484, 81.3383484
3: -28.4008694, 59.6408424, -28.4008694, 59.6408424, -88.0417099, 88.0417099
4: -21.3142948, 59.0830536, -21.3142948, 59.0830536, -80.3973465, 80.3973465

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.7374048, 69.2293091, -79.6738129, 68.7017746
1: -16.9038010, 66.4907379, -20.6976452, 82.3365021, -99.2402802, 87.1883774
2: -14.7949934, 66.5433578, -17.9375267, 82.3648682, -97.1598587, 84.4808655
3: -28.4008694, 59.6408424, -34.9264603, 73.7202835, -102.1211548, 94.5673065
4: -21.3142948, 59.0830536, -26.1118031, 72.8422928, -94.1565857, 85.1948547

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -10.4445028, 55.9643669, -68.7017670, 79.6738052
1: -20.6976452, 82.3365021, -16.9038010, 66.4907379, -87.1883774, 99.2402878
2: -17.9375267, 82.3648682, -14.7949934, 66.5433578, -84.4808731, 97.1598587
3: -34.9264603, 73.7202835, -28.4008694, 59.6408424, -94.5672989, 102.1211548
4: -26.1118031, 72.8422928, -21.3142948, 59.0830536, -85.1948547, 94.1565857

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.7374048, 69.2293091, -81.9667130, 81.9667130
1: -20.6976452, 82.3365021, -20.6976452, 82.3365021, -103.0341263, 103.0341339
2: -17.9375267, 82.3648682, -17.9375267, 82.3648682, -100.3023834, 100.3023834
3: -34.9264603, 73.7202835, -34.9264603, 73.7202835, -108.6467438, 108.6467438
4: -26.1118031, 72.8422928, -26.1118031, 72.8422928, -98.9540863, 98.9540939

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -14.6269550, 82.7033157, -93.1478195, 70.5913239
1: -16.9038010, 66.4907379, -23.8081474, 98.2509613, -115.1547546, 90.2988892
2: -14.7949934, 66.5433578, -20.6988335, 98.1297531, -112.9247437, 87.2421646
3: -28.4008694, 59.6408424, -40.2802467, 87.0169601, -115.4178314, 99.9210739
4: -21.3142948, 59.0830536, -30.1702652, 85.8516464, -107.1659393, 89.2533188

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4962782, upper bound: 112.4732156
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.65 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.64 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.64
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.64
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.64
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.64
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.64
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.64
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.64
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -10.4445028, 55.9643669, -66.0205383, 64.4386826
1: -16.2604694, 64.1127777, -16.9038010, 66.4907379, -82.7512054, 81.0165634
2: -14.2520180, 64.1702576, -14.7949934, 66.5433578, -80.7953568, 78.9652481
3: -27.2941570, 57.4110603, -28.4008694, 59.6408424, -86.9349976, 85.8119278
4: -20.4760780, 56.9184914, -21.3142948, 59.0830536, -79.5591202, 78.2327881

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -12.7374048, 69.2293091, -79.2854767, 66.7315750
1: -16.2604694, 64.1127777, -20.6976452, 82.3365021, -98.5969620, 84.8104095
2: -14.2520180, 64.1702576, -17.9375267, 82.3648682, -96.6168671, 82.1077805
3: -27.2941570, 57.4110603, -34.9264603, 73.7202835, -101.0144424, 92.3375092
4: -20.4760780, 56.9184914, -26.1118031, 72.8422928, -93.3183670, 83.0302811

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -10.4445028, 55.9643669, -68.2222595, 77.2292023
1: -19.9063568, 79.3842850, -16.9038010, 66.4907379, -86.3970947, 96.2880783
2: -17.2704391, 79.4186096, -14.7949934, 66.5433578, -83.8137894, 94.2136002
3: -33.5641937, 70.9576569, -28.4008694, 59.6408424, -93.2050323, 99.3585281
4: -25.0775681, 70.1679382, -21.3142948, 59.0830536, -84.1606216, 91.4822311

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -12.7374048, 69.2293091, -81.4871902, 79.5221100
1: -19.9063568, 79.3842850, -20.6976452, 82.3365021, -102.2428513, 100.0819244
2: -17.2704391, 79.4186096, -17.9375267, 82.3648682, -99.6352997, 97.3561249
3: -33.5641937, 70.9576569, -34.9264603, 73.7202835, -107.2844772, 105.8841095
4: -25.0775681, 70.1679382, -26.1118031, 72.8422928, -97.9198608, 96.2797394

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.07 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.07
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
Binary search (step 11): status=Status.VERIFIED, low=0.1249695, high=0.1250000, mid=0.1249695, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 12) starts
Candidate diff: 0.1249847


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.72 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.62 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.62
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -14.8509598, 83.8702927, -96.7212601, 84.5706635
1: -20.9004955, 82.9266129, -24.1805515, 99.6739273, -120.5744247, 107.1071625
2: -18.1388531, 82.9589615, -20.9075127, 99.5991058, -117.7379608, 103.8664551
3: -35.2660179, 74.2744598, -40.9821777, 88.5371628, -123.8031769, 115.2566376
4: -26.3749428, 73.4100494, -30.6739483, 87.2144394, -113.5893555, 104.0839996

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.64 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -14.8509598, 83.8702927, -99.0249863, 100.3664780
1: -24.6804409, 101.6211853, -24.1805515, 99.6739273, -124.3543701, 125.8017349
2: -21.4360867, 101.5054321, -20.9075127, 99.5991058, -121.0351868, 122.4129333
3: -41.7919083, 90.1225281, -40.9821777, 88.5371628, -130.3290405, 131.1046753
4: -31.3080330, 88.8759460, -30.6739483, 87.2144394, -118.5224609, 119.5498962

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.75 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.65 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.65
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -12.8509674, 69.7197113, -82.5706787, 82.5706787
1: -20.9004955, 82.9266129, -20.9004955, 82.9266129, -103.8271103, 103.8271103
2: -18.1388531, 82.9589615, -18.1388531, 82.9589615, -101.0978165, 101.0978165
3: -35.2660179, 74.2744598, -35.2660179, 74.2744598, -109.5404816, 109.5404816
4: -26.3749428, 73.4100494, -26.3749428, 73.4100494, -99.7849884, 99.7849884

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.63 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -15.1547003, 85.5155258, -98.3664932, 84.8744049
1: -20.9004955, 82.9266129, -24.6804409, 101.6211853, -122.5216827, 107.6070557
2: -18.1388531, 82.9589615, -21.4360867, 101.5054321, -119.6442871, 104.3950500
3: -35.2660179, 74.2744598, -41.7919083, 90.1225281, -125.3885498, 116.0663605
4: -26.3749428, 73.4100494, -31.3080330, 88.8759460, -115.2508850, 104.7180786

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.71 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -12.8509674, 69.7197113, -84.8744049, 98.3664932
1: -24.6804409, 101.6211853, -20.9004955, 82.9266129, -107.6070557, 122.5216827
2: -21.4360867, 101.5054321, -18.1388531, 82.9589615, -104.3950500, 119.6442871
3: -41.7919083, 90.1225281, -35.2660179, 74.2744598, -116.0663605, 125.3885498
4: -31.3080330, 88.8759460, -26.3749428, 73.4100494, -104.7180786, 115.2508850

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.77 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -15.1547003, 85.5155258, -100.6702194, 100.6702271
1: -24.6804409, 101.6211853, -24.6804409, 101.6211853, -126.3016281, 126.3016281
2: -21.4360867, 101.5054321, -21.4360867, 101.5054321, -122.9415207, 122.9415207
3: -41.7919083, 90.1225281, -41.7919083, 90.1225281, -131.9143982, 131.9143982
4: -31.3080330, 88.8759460, -31.3080330, 88.8759460, -120.1839752, 120.1839752

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.56 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.39 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.39
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.39
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.39
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.39
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.39
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.8509674, 69.7197113, -80.1642075, 68.8153381
1: -16.9038010, 66.4907379, -20.9004955, 82.9266129, -99.8304062, 87.3912354
2: -14.7949934, 66.5433578, -18.1388531, 82.9589615, -97.7539520, 84.6822128
3: -28.4008694, 59.6408424, -35.2660179, 74.2744598, -102.6753311, 94.9068604
4: -21.3142948, 59.0830536, -26.3749428, 73.4100494, -94.7243423, 85.4579849

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.8509674, 69.7197113, -82.4571075, 82.0802765
1: -20.6976452, 82.3365021, -20.9004955, 82.9266129, -103.6242523, 103.2369919
2: -17.9375267, 82.3648682, -18.1388531, 82.9589615, -100.8964767, 100.5037231
3: -34.9264603, 73.7202835, -35.2660179, 74.2744598, -109.2009201, 108.9862976
4: -26.1118031, 72.8422928, -26.3749428, 73.4100494, -99.5218430, 99.2172318

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -15.1547003, 85.5155258, -95.9600296, 71.1190643
1: -16.9038010, 66.4907379, -24.6804409, 101.6211853, -118.5249786, 91.1711807
2: -14.7949934, 66.5433578, -21.4360867, 101.5054321, -116.3004227, 87.9794464
3: -28.4008694, 59.6408424, -41.7919083, 90.1225281, -118.5233994, 101.4327393
4: -21.3142948, 59.0830536, -31.3080330, 88.8759460, -110.1902390, 90.3910828

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5405044, upper bound: 112.4936470
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -15.1547003, 85.5155258, -98.2529221, 84.3840027
1: -20.6976452, 82.3365021, -24.6804409, 101.6211853, -122.3188324, 107.0169449
2: -17.9375267, 82.3648682, -21.4360867, 101.5054321, -119.4429474, 103.8009567
3: -34.9264603, 73.7202835, -41.7919083, 90.1225281, -125.0489883, 115.5121918
4: -26.1118031, 72.8422928, -31.3080330, 88.8759460, -114.9877472, 104.1503296

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295
time: 0.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.98 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.98
Output dim: 3, lower bound: -112.5405044, upper bound: 112.4936470
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.98
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.98
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.98
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -10.4445028, 55.9643669, -66.4088669, 66.4088593
1: -16.9038010, 66.4907379, -16.9038010, 66.4907379, -83.3945312, 83.3945312
2: -14.7949934, 66.5433578, -14.7949934, 66.5433578, -81.3383484, 81.3383484
3: -28.4008694, 59.6408424, -28.4008694, 59.6408424, -88.0417099, 88.0417099
4: -21.3142948, 59.0830536, -21.3142948, 59.0830536, -80.3973465, 80.3973465

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.7374048, 69.2293091, -79.6738129, 68.7017746
1: -16.9038010, 66.4907379, -20.6976452, 82.3365021, -99.2402802, 87.1883774
2: -14.7949934, 66.5433578, -17.9375267, 82.3648682, -97.1598587, 84.4808655
3: -28.4008694, 59.6408424, -34.9264603, 73.7202835, -102.1211548, 94.5673065
4: -21.3142948, 59.0830536, -26.1118031, 72.8422928, -94.1565857, 85.1948547

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.92 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -10.4445028, 55.9643669, -68.7017670, 79.6738052
1: -20.6976452, 82.3365021, -16.9038010, 66.4907379, -87.1883774, 99.2402878
2: -17.9375267, 82.3648682, -14.7949934, 66.5433578, -84.4808731, 97.1598587
3: -34.9264603, 73.7202835, -28.4008694, 59.6408424, -94.5672989, 102.1211548
4: -26.1118031, 72.8422928, -21.3142948, 59.0830536, -85.1948547, 94.1565857

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.7374048, 69.2293091, -81.9667130, 81.9667130
1: -20.6976452, 82.3365021, -20.6976452, 82.3365021, -103.0341263, 103.0341339
2: -17.9375267, 82.3648682, -17.9375267, 82.3648682, -100.3023834, 100.3023834
3: -34.9264603, 73.7202835, -34.9264603, 73.7202835, -108.6467438, 108.6467438
4: -26.1118031, 72.8422928, -26.1118031, 72.8422928, -98.9540863, 98.9540939

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -14.6269550, 82.7033157, -93.1478195, 70.5913239
1: -16.9038010, 66.4907379, -23.8081474, 98.2509613, -115.1547546, 90.2988892
2: -14.7949934, 66.5433578, -20.6988335, 98.1297531, -112.9247437, 87.2421646
3: -28.4008694, 59.6408424, -40.2802467, 87.0169601, -115.4178314, 99.9210739
4: -21.3142948, 59.0830536, -30.1702652, 85.8516464, -107.1659393, 89.2533188

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4962782, upper bound: 112.4732156
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.64 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.67 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.67
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.67
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.67
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.67
Output dim: 3, lower bound: -112.4830340, upper bound: 112.5465856
IS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.67
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.67
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.67
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -10.4445028, 55.9643669, -66.0205383, 64.4386826
1: -16.2604694, 64.1127777, -16.9038010, 66.4907379, -82.7512054, 81.0165634
2: -14.2520180, 64.1702576, -14.7949934, 66.5433578, -80.7953568, 78.9652481
3: -27.2941570, 57.4110603, -28.4008694, 59.6408424, -86.9349976, 85.8119278
4: -20.4760780, 56.9184914, -21.3142948, 59.0830536, -79.5591202, 78.2327881

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.0561686, 53.9941826, -12.7374048, 69.2293091, -79.2854767, 66.7315750
1: -16.2604694, 64.1127777, -20.6976452, 82.3365021, -98.5969620, 84.8104095
2: -14.2520180, 64.1702576, -17.9375267, 82.3648682, -96.6168671, 82.1077805
3: -27.2941570, 57.4110603, -34.9264603, 73.7202835, -101.0144424, 92.3375092
4: -20.4760780, 56.9184914, -26.1118031, 72.8422928, -93.3183670, 83.0302811

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -10.4445028, 55.9643669, -68.2222595, 77.2292023
1: -19.9063568, 79.3842850, -16.9038010, 66.4907379, -86.3970947, 96.2880783
2: -17.2704391, 79.4186096, -14.7949934, 66.5433578, -83.8137894, 94.2136002
3: -33.5641937, 70.9576569, -28.4008694, 59.6408424, -93.2050323, 99.3585281
4: -25.0775681, 70.1679382, -21.3142948, 59.0830536, -84.1606216, 91.4822311

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -12.2578955, 66.7847061, -12.7374048, 69.2293091, -81.4871902, 79.5221100
1: -19.9063568, 79.3842850, -20.6976452, 82.3365021, -102.2428513, 100.0819244
2: -17.2704391, 79.4186096, -17.9375267, 82.3648682, -99.6352997, 97.3561249
3: -33.5641937, 70.9576569, -34.9264603, 73.7202835, -107.2844772, 105.8841095
4: -25.0775681, 70.1679382, -26.1118031, 72.8422928, -97.9198608, 96.2797394

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
time: 0.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.29 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.29
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.29
Output dim: 3, lower bound: -112.4808694, upper bound: 112.4808694
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.29
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.29
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.29
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.29
Output dim: 3, lower bound: -112.4645241, upper bound: 112.4584667
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.29
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.29
Output dim: 3, lower bound: -112.4421215, upper bound: 112.4421215
Binary search (step 12): status=Status.VERIFIED, low=0.1249847, high=0.1250000, mid=0.1249847, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 13) starts
Candidate diff: 0.1249924


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.75 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.70 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.70
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -14.8509598, 83.8702927, -96.7212601, 84.5706635
1: -20.9004955, 82.9266129, -24.1805515, 99.6739273, -120.5744247, 107.1071625
2: -18.1388531, 82.9589615, -20.9075127, 99.5991058, -117.7379608, 103.8664551
3: -35.2660179, 74.2744598, -40.9821777, 88.5371628, -123.8031769, 115.2566376
4: -26.3749428, 73.4100494, -30.6739483, 87.2144394, -113.5893555, 104.0839996

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.74 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -14.8509598, 83.8702927, -99.0249863, 100.3664780
1: -24.6804409, 101.6211853, -24.1805515, 99.6739273, -124.3543701, 125.8017349
2: -21.4360867, 101.5054321, -20.9075127, 99.5991058, -121.0351868, 122.4129333
3: -41.7919083, 90.1225281, -40.9821777, 88.5371628, -130.3290405, 131.1046753
4: -31.3080330, 88.8759460, -30.6739483, 87.2144394, -118.5224609, 119.5498962

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.66 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
time: 0.61 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.44 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.44
Output dim: 3, lower bound: -112.5536375, upper bound: 112.5536375

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -12.8509674, 69.7197113, -82.5706787, 82.5706787
1: -20.9004955, 82.9266129, -20.9004955, 82.9266129, -103.8271103, 103.8271103
2: -18.1388531, 82.9589615, -18.1388531, 82.9589615, -101.0978165, 101.0978165
3: -35.2660179, 74.2744598, -35.2660179, 74.2744598, -109.5404816, 109.5404816
4: -26.3749428, 73.4100494, -26.3749428, 73.4100494, -99.7849884, 99.7849884

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.72 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -12.8509674, 69.7197113, -15.1547003, 85.5155258, -98.3664932, 84.8744049
1: -20.9004955, 82.9266129, -24.6804409, 101.6211853, -122.5216827, 107.6070557
2: -18.1388531, 82.9589615, -21.4360867, 101.5054321, -119.6442871, 104.3950500
3: -35.2660179, 74.2744598, -41.7919083, 90.1225281, -125.3885498, 116.0663605
4: -26.3749428, 73.4100494, -31.3080330, 88.8759460, -115.2508850, 104.7180786

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
time: 0.85 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -12.8509674, 69.7197113, -84.8744049, 98.3664932
1: -24.6804409, 101.6211853, -20.9004955, 82.9266129, -107.6070557, 122.5216827
2: -21.4360867, 101.5054321, -18.1388531, 82.9589615, -104.3950500, 119.6442871
3: -41.7919083, 90.1225281, -35.2660179, 74.2744598, -116.0663605, 125.3885498
4: -31.3080330, 88.8759460, -26.3749428, 73.4100494, -104.7180786, 115.2508850

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.77 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -15.1547003, 85.5155258, -15.1547003, 85.5155258, -100.6702194, 100.6702271
1: -24.6804409, 101.6211853, -24.6804409, 101.6211853, -126.3016281, 126.3016281
2: -21.4360867, 101.5054321, -21.4360867, 101.5054321, -122.9415207, 122.9415207
3: -41.7919083, 90.1225281, -41.7919083, 90.1225281, -131.9143982, 131.9143982
4: -31.3080330, 88.8759460, -31.3080330, 88.8759460, -120.1839752, 120.1839752

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of IS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
time: 0.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.46 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.5868088, upper bound: 112.6179648
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.5654577, upper bound: 112.5760830
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.4355923, upper bound: 112.5008673
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 4.46
Output dim: 3, lower bound: -112.3831776, upper bound: 112.3831776

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.8509674, 69.7197113, -80.1642075, 68.8153381
1: -16.9038010, 66.4907379, -20.9004955, 82.9266129, -99.8304062, 87.3912354
2: -14.7949934, 66.5433578, -18.1388531, 82.9589615, -97.7539520, 84.6822128
3: -28.4008694, 59.6408424, -35.2660179, 74.2744598, -102.6753311, 94.9068604
4: -21.3142948, 59.0830536, -26.3749428, 73.4100494, -94.7243423, 85.4579849

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -12.8509674, 69.7197113, -82.4571075, 82.0802765
1: -20.6976452, 82.3365021, -20.9004955, 82.9266129, -103.6242523, 103.2369919
2: -17.9375267, 82.3648682, -18.1388531, 82.9589615, -100.8964767, 100.5037231
3: -34.9264603, 73.7202835, -35.2660179, 74.2744598, -109.2009201, 108.9862976
4: -26.1118031, 72.8422928, -26.3749428, 73.4100494, -99.5218430, 99.2172318

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -15.1547003, 85.5155258, -95.9600296, 71.1190643
1: -16.9038010, 66.4907379, -24.6804409, 101.6211853, -118.5249786, 91.1711807
2: -14.7949934, 66.5433578, -21.4360867, 101.5054321, -116.3004227, 87.9794464
3: -28.4008694, 59.6408424, -41.7919083, 90.1225281, -118.5233994, 101.4327393
4: -21.3142948, 59.0830536, -31.3080330, 88.8759460, -110.1902390, 90.3910828

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5405045, upper bound: 112.4936470
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -12.7374048, 69.2293091, -15.1547003, 85.5155258, -98.2529221, 84.3840027
1: -20.6976452, 82.3365021, -24.6804409, 101.6211853, -122.3188324, 107.0169449
2: -17.9375267, 82.3648682, -21.4360867, 101.5054321, -119.4429474, 103.8009567
3: -34.9264603, 73.7202835, -41.7919083, 90.1225281, -125.0489883, 115.5121918
4: -26.1118031, 72.8422928, -31.3080330, 88.8759460, -114.9877472, 104.1503296

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 46

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295
time: 0.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.70 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 3, lower bound: -112.5879032, upper bound: 112.5879032
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.70
Output dim: 3, lower bound: -112.5405045, upper bound: 112.4936470
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.70
Output dim: 3, lower bound: -112.4228147, upper bound: 112.4412322
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 4.70
Output dim: 3, lower bound: -112.5241592, upper bound: 112.4712443
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 4.70
Output dim: 3, lower bound: -112.4064695, upper bound: 112.4188295

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -10.4445028, 55.9643669, -66.4088669, 66.4088593
1: -16.9038010, 66.4907379, -16.9038010, 66.4907379, -83.3945312, 83.3945312
2: -14.7949934, 66.5433578, -14.7949934, 66.5433578, -81.3383484, 81.3383484
3: -28.4008694, 59.6408424, -28.4008694, 59.6408424, -88.0417099, 88.0417099
4: -21.3142948, 59.0830536, -21.3142948, 59.0830536, -80.3973465, 80.3973465

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of IS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5044040, upper bound: 112.5887287
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4584667, upper bound: 112.4645241
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -10.4445028, 55.9643669, -12.7374048, 69.2293091, -79.6738129, 68.7017746
1: -16.9038010, 66.4907379, -20.6976452, 82.3365021, -99.2402802, 87.1883774
2: -14.7949934, 66.5433578, -17.9375267, 82.3648682, -97.1598587, 84.4808655
3: -28.4008694, 59.6408424, -34.9264603, 73.7202835, -102.1211548, 94.5673065
4: -21.3142948, 59.0830536, -26.1118031, 72.8422928, -94.1565857, 85.1948547

Time for backsubstitution: 2.06 seconds
Binary search (step 13): status=Status.UNKNOWN, low=0.1249847, high=0.1249924, mid=0.1249924, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.1249847412109375
execution time: 1131.71 seconds
