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
execution time: IAR + LP analysis = 2.08 + 1.74 = 3.82 seconds
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
Binary search time: 65.52 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 1130.66 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4851123, upper bound: 112.4851123
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4851123, upper bound: 112.4851123
time: 0.64 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.29 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.29
Output dim: 3, lower bound: -112.4851123, upper bound: 112.4851123
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.29
Output dim: 3, lower bound: -112.4851123, upper bound: 112.4851123
Binary search (step 0): status=Status.VERIFIED, low=0.0625000, high=0.1250000, mid=0.0625000, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072838]}

## Binary search (step 1) starts
Candidate diff: 0.0937500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6461283, upper bound: 112.6461283
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6461283, upper bound: 112.6461283
time: 0.80 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.68 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 3, lower bound: -112.6461283, upper bound: 112.6461283
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.68
Output dim: 3, lower bound: -112.6461283, upper bound: 112.6461283

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6458313, upper bound: 112.6458305
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6458305, upper bound: 112.6458313
time: 0.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6305637, upper bound: 112.6305637
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6305637, upper bound: 112.6305637
time: 0.58 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.09 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 3, lower bound: -112.6458313, upper bound: 112.6458305
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 3, lower bound: -112.6458305, upper bound: 112.6458313
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 3, lower bound: -112.6305637, upper bound: 112.6305637
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 3, lower bound: -112.6305637, upper bound: 112.6305637

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4519004, upper bound: 112.4522204
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4519004, upper bound: 112.4522204
time: 0.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6458305, upper bound: 112.6457245
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6458205, upper bound: 112.6458313
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5694091, upper bound: 112.5694084
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5694091, upper bound: 112.5694084
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6232593, upper bound: 112.6232493
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6232593, upper bound: 112.6232493
time: 0.66 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.25
Output dim: 3, lower bound: -112.4519004, upper bound: 112.4522204
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.25
Output dim: 3, lower bound: -112.4519004, upper bound: 112.4522204
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 3, lower bound: -112.6458305, upper bound: 112.6457245
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 3, lower bound: -112.6458205, upper bound: 112.6458313
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 3, lower bound: -112.5694091, upper bound: 112.5694084
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 3, lower bound: -112.5694091, upper bound: 112.5694084
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 3, lower bound: -112.6232593, upper bound: 112.6232493
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.25
Output dim: 3, lower bound: -112.6232593, upper bound: 112.6232493

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3817212, upper bound: 112.3817212
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3817212, upper bound: 112.3817212
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5688676, upper bound: 112.5688676
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5688676, upper bound: 112.5688676
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5226068, upper bound: 112.5226559
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5226068, upper bound: 112.5226559
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5679380, upper bound: 112.5679380
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5679626, upper bound: 112.5679380
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6228100, upper bound: 112.6230358
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6230506, upper bound: 112.6228100
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4835445, upper bound: 112.4835445
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4835644, upper bound: 112.4835445
time: 0.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.12 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 3, lower bound: -112.3817212, upper bound: 112.3817212
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 3, lower bound: -112.3817212, upper bound: 112.3817212
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -112.5688676, upper bound: 112.5688676
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -112.5688676, upper bound: 112.5688676
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 3, lower bound: -112.5226068, upper bound: 112.5226559
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 3, lower bound: -112.5226068, upper bound: 112.5226559
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -112.5679380, upper bound: 112.5679380
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -112.5679626, upper bound: 112.5679380
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -112.6228100, upper bound: 112.6230358
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.12
Output dim: 3, lower bound: -112.6230506, upper bound: 112.6228100
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 3, lower bound: -112.4835445, upper bound: 112.4835445
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.12
Output dim: 3, lower bound: -112.4835644, upper bound: 112.4835445

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4522204, upper bound: 112.4519004
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4522204, upper bound: 112.4519004
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5689451, upper bound: 112.5688676
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5688676, upper bound: 112.5688676
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.1833054, upper bound: 112.1833054
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.1833054, upper bound: 112.1833054
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5674167, upper bound: 112.5674167
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5674167, upper bound: 112.5674167
time: 0.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5576065, upper bound: 112.5576065
time: 0.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5576065, upper bound: 112.5576065
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4847353, upper bound: 112.4835445
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4847353, upper bound: 112.4835445
time: 0.58 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.21 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.4522204, upper bound: 112.4519004
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.4522204, upper bound: 112.4519004
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.5689451, upper bound: 112.5688676
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.5688676, upper bound: 112.5688676
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.1833054, upper bound: 112.1833054
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.1833054, upper bound: 112.1833054
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.5674167, upper bound: 112.5674167
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.5674167, upper bound: 112.5674167
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.5576065, upper bound: 112.5576065
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.5576065, upper bound: 112.5576065
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.4847353, upper bound: 112.4835445
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.4847353, upper bound: 112.4835445

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3817212, upper bound: 112.3817212
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3817212, upper bound: 112.3817212
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5685391, upper bound: 112.5685391
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5685391, upper bound: 112.5685391
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5209303, upper bound: 112.5209303
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5209303, upper bound: 112.5209303
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5674252, upper bound: 112.5674167
time: 0.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5675053, upper bound: 112.5674167
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5576065, upper bound: 112.5576065
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5576065, upper bound: 112.5576065
time: 0.59 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.46 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.46
Output dim: 3, lower bound: -112.3817212, upper bound: 112.3817212
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.46
Output dim: 3, lower bound: -112.3817212, upper bound: 112.3817212
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 3, lower bound: -112.5685391, upper bound: 112.5685391
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 3, lower bound: -112.5685391, upper bound: 112.5685391
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.46
Output dim: 3, lower bound: -112.5209303, upper bound: 112.5209303
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.46
Output dim: 3, lower bound: -112.5209303, upper bound: 112.5209303
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 3, lower bound: -112.5674252, upper bound: 112.5674167
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 3, lower bound: -112.5675053, upper bound: 112.5674167
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 3, lower bound: -112.5576065, upper bound: 112.5576065
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.46
Output dim: 3, lower bound: -112.5576065, upper bound: 112.5576065

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5685391, upper bound: 112.5685391
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5685391, upper bound: 112.5685391
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5670923, upper bound: 112.5670923
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5670923, upper bound: 112.5670923
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5209303, upper bound: 112.5209303
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5209303, upper bound: 112.5209303
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5675053, upper bound: 112.5674167
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5674284, upper bound: 112.5674167
time: 0.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5576065, upper bound: 112.5576065
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5576065, upper bound: 112.5576065
time: 0.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.63 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.72 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5685391, upper bound: 112.5685391
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5685391, upper bound: 112.5685391
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5670923, upper bound: 112.5670923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5670923, upper bound: 112.5670923
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5209303, upper bound: 112.5209303
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5209303, upper bound: 112.5209303
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5675053, upper bound: 112.5674167
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5674284, upper bound: 112.5674167
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5576065, upper bound: 112.5576065
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5576065, upper bound: 112.5576065
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.72
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5219656, upper bound: 112.5219656
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5219656, upper bound: 112.5219656
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5677644, upper bound: 112.5677644
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5677644, upper bound: 112.5677644
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5670923, upper bound: 112.5670923
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5670923, upper bound: 112.5670923
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5663437, upper bound: 112.5663437
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5663437, upper bound: 112.5663437
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.2147969, upper bound: 112.2147969
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.2147969, upper bound: 112.2147969
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5209303, upper bound: 112.5209303
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5209303, upper bound: 112.5209303
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 1.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.75 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 3.55 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5219656, upper bound: 112.5219656
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5219656, upper bound: 112.5219656
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5677644, upper bound: 112.5677644
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5677644, upper bound: 112.5677644
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5670923, upper bound: 112.5670923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5670923, upper bound: 112.5670923
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5663437, upper bound: 112.5663437
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5663437, upper bound: 112.5663437
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.2147969, upper bound: 112.2147969
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.2147969, upper bound: 112.2147969
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5209303, upper bound: 112.5209303
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5209303, upper bound: 112.5209303
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 3.55
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5211643, upper bound: 112.5211643
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5211643, upper bound: 112.5211643
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3704258, upper bound: 112.3704258
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3704258, upper bound: 112.3704258
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.2121480, upper bound: 112.2121480
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.2121480, upper bound: 112.2121480
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.2121480, upper bound: 112.2121480
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.2121480, upper bound: 112.2121480
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4282325, upper bound: 112.4282325
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4282325, upper bound: 112.4282325
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.1893528, upper bound: 112.1893528
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.1893528, upper bound: 112.1893528
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.75 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 4.77 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5211643, upper bound: 112.5211643
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5211643, upper bound: 112.5211643
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.3704258, upper bound: 112.3704258
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.3704258, upper bound: 112.3704258
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.2121480, upper bound: 112.2121480
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.2121480, upper bound: 112.2121480
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.2121480, upper bound: 112.2121480
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.2121480, upper bound: 112.2121480
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.4282325, upper bound: 112.4282325
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.4282325, upper bound: 112.4282325
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.1893528, upper bound: 112.1893528
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.1893528, upper bound: 112.1893528
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.77
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 3.70 seconds

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 3.77 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 33

Time for candidate selection: 3.80 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 22

Time for candidate selection: 3.78 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 33

Time for candidate selection: 3.74 seconds

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 3.78 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 3.80 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 3.77 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5095036, upper bound: 112.5095036
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5095036, upper bound: 112.5095036
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 3.77 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 3.87 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 3.90 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 3.87 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 3.78 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 3.96 seconds

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 3.88 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 3.85 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.18 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 3.91 seconds

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 33

Time for candidate selection: 3.95 seconds

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.
Binary search (step 1): status=Status.UNKNOWN, low=0.0625000, high=0.0937500, mid=0.0937500, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072838, 112.65165179072838]}

## Binary search (step 2) starts
Candidate diff: 0.0781250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4851123, upper bound: 112.4851123
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4851123, upper bound: 112.4851123
time: 0.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.23 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.23
Output dim: 3, lower bound: -112.4851123, upper bound: 112.4851123
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.23
Output dim: 3, lower bound: -112.4851123, upper bound: 112.4851123
Binary search (step 2): status=Status.VERIFIED, low=0.0781250, high=0.0937500, mid=0.0781250, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 3) starts
Candidate diff: 0.0859375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6516141, upper bound: 112.6515172
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6515172, upper bound: 112.6516141
time: 0.66 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 3, lower bound: -112.6516141, upper bound: 112.6515172
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 3, lower bound: -112.6515172, upper bound: 112.6516141

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3247913, upper bound: 112.3247913
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3247913, upper bound: 112.3247913
time: 0.78 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6502959, upper bound: 112.6516141
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6502959, upper bound: 112.6511526
time: 0.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.30 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.30
Output dim: 3, lower bound: -112.3247913, upper bound: 112.3247913
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.30
Output dim: 3, lower bound: -112.3247913, upper bound: 112.3247913
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 3, lower bound: -112.6502959, upper bound: 112.6516141
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.30
Output dim: 3, lower bound: -112.6502959, upper bound: 112.6511526

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4621414, upper bound: 112.4621414
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4621414, upper bound: 112.4621414
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512602, upper bound: 112.6502377
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512117, upper bound: 112.6508753
time: 0.83 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.56 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.56
Output dim: 3, lower bound: -112.4621414, upper bound: 112.4621414
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.56
Output dim: 3, lower bound: -112.4621414, upper bound: 112.4621414
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 3, lower bound: -112.6512602, upper bound: 112.6502377
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.56
Output dim: 3, lower bound: -112.6512117, upper bound: 112.6508753

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3240773, upper bound: 112.3240773
time: 0.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3240773, upper bound: 112.3240773
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4345647, upper bound: 112.4345647
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4345647, upper bound: 112.4345647
time: 0.61 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.31 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 3, lower bound: -112.3240773, upper bound: 112.3240773
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 3, lower bound: -112.3240773, upper bound: 112.3240773
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 3, lower bound: -112.4345647, upper bound: 112.4345647
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.31
Output dim: 3, lower bound: -112.4345647, upper bound: 112.4345647
Binary search (step 3): status=Status.VERIFIED, low=0.0859375, high=0.0937500, mid=0.0859375, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072838, 112.65165179072838]}

## Binary search (step 4) starts
Candidate diff: 0.0898438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6379278, upper bound: 112.6379278
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6379278, upper bound: 112.6379278
time: 0.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 3, lower bound: -112.6379278, upper bound: 112.6379278
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.58
Output dim: 3, lower bound: -112.6379278, upper bound: 112.6379278

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5269481, upper bound: 112.5269481
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5269481, upper bound: 112.5269481
time: 0.79 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5269481, upper bound: 112.5269481
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5269481, upper bound: 112.5269481
time: 0.78 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.92 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.92
Output dim: 3, lower bound: -112.5269481, upper bound: 112.5269481
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.92
Output dim: 3, lower bound: -112.5269481, upper bound: 112.5269481
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.92
Output dim: 3, lower bound: -112.5269481, upper bound: 112.5269481
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.92
Output dim: 3, lower bound: -112.5269481, upper bound: 112.5269481
Binary search (step 4): status=Status.VERIFIED, low=0.0898438, high=0.0937500, mid=0.0898438, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072838, 112.65165179072838]}

## Binary search (step 5) starts
Candidate diff: 0.0917969


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5269481, upper bound: 112.5269481
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5269481, upper bound: 112.5269481
time: 0.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.48 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.48
Output dim: 3, lower bound: -112.5269481, upper bound: 112.5269481
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.48
Output dim: 3, lower bound: -112.5269481, upper bound: 112.5269481
Binary search (step 5): status=Status.VERIFIED, low=0.0917969, high=0.0937500, mid=0.0917969, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072838]}

## Binary search (step 6) starts
Candidate diff: 0.0927734


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3634801, upper bound: 112.3634801
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3634801, upper bound: 112.3634801
time: 0.61 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.23 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 1.23
Output dim: 3, lower bound: -112.3634801, upper bound: 112.3634801
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 1.23
Output dim: 3, lower bound: -112.3634801, upper bound: 112.3634801
Binary search (step 6): status=Status.VERIFIED, low=0.0927734, high=0.0937500, mid=0.0927734, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072838, 112.65165179072838]}

## Binary search (step 7) starts
Candidate diff: 0.0932617


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6516141, upper bound: 112.6515172
time: 0.94 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6515172, upper bound: 112.6516141
time: 0.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.67 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 3, lower bound: -112.6516141, upper bound: 112.6515172
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.67
Output dim: 3, lower bound: -112.6515172, upper bound: 112.6516141

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4621414, upper bound: 112.4621414
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4621414, upper bound: 112.4621414
time: 0.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6503276, upper bound: 112.6516141
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6515172, upper bound: 112.6508223
time: 0.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.58 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.58
Output dim: 3, lower bound: -112.4621414, upper bound: 112.4621414
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.58
Output dim: 3, lower bound: -112.4621414, upper bound: 112.4621414
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 3, lower bound: -112.6503276, upper bound: 112.6516141
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.58
Output dim: 3, lower bound: -112.6515172, upper bound: 112.6508223

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6459505, upper bound: 112.6460876
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6459505, upper bound: 112.6460876
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6460111, upper bound: 112.6460874
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6460111, upper bound: 112.6460874
time: 0.67 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.28 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 3, lower bound: -112.6459505, upper bound: 112.6460876
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 3, lower bound: -112.6459505, upper bound: 112.6460876
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 3, lower bound: -112.6460111, upper bound: 112.6460874
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 3, lower bound: -112.6460111, upper bound: 112.6460874

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.1936352, upper bound: 112.1936352
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.1936352, upper bound: 112.1936352
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5203534, upper bound: 112.5203593
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5203534, upper bound: 112.5203593
time: 0.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6460111, upper bound: 112.6460874
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6459505, upper bound: 112.6460225
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4330129, upper bound: 112.4330129
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4330129, upper bound: 112.4330129
time: 0.60 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.25 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 3, lower bound: -112.1936352, upper bound: 112.1936352
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 3, lower bound: -112.1936352, upper bound: 112.1936352
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 3, lower bound: -112.5203534, upper bound: 112.5203593
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 3, lower bound: -112.5203534, upper bound: 112.5203593
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 3, lower bound: -112.6460111, upper bound: 112.6460874
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.25
Output dim: 3, lower bound: -112.6459505, upper bound: 112.6460225
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 3, lower bound: -112.4330129, upper bound: 112.4330129
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.25
Output dim: 3, lower bound: -112.4330129, upper bound: 112.4330129

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6303778, upper bound: 112.6302996
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6303778, upper bound: 112.6302996
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4319765, upper bound: 112.4319765
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4319765, upper bound: 112.4319765
time: 0.64 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.21 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.6303778, upper bound: 112.6302996
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.6303778, upper bound: 112.6302996
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.4319765, upper bound: 112.4319765
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.21
Output dim: 3, lower bound: -112.4319765, upper bound: 112.4319765

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4282325, upper bound: 112.4319354
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4282325, upper bound: 112.4319354
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575550, upper bound: 112.5575550
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575550, upper bound: 112.5575550
time: 0.72 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.74 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.74
Output dim: 3, lower bound: -112.4282325, upper bound: 112.4319354
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.74
Output dim: 3, lower bound: -112.4282325, upper bound: 112.4319354
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 3, lower bound: -112.5575550, upper bound: 112.5575550
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.74
Output dim: 3, lower bound: -112.5575550, upper bound: 112.5575550

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575550, upper bound: 112.5575550
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575550, upper bound: 112.5575550
time: 0.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.53 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.50 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -112.5575550, upper bound: 112.5575550
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -112.5575550, upper bound: 112.5575550
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 3.50
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.59 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.58 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.58
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.58
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.58
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.58
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.58
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.58
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.58
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.58
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.62 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.81 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 4.36 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 9, time: 4.36
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 3.60 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 3.74 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 3.69 seconds

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 3.66 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 3.60 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 3.65 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 3.76 seconds

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 3.69 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 3.62 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 21

Time for candidate selection: 3.71 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 3.73 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 33

Time for candidate selection: 3.73 seconds

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 3.64 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 21

Time for candidate selection: 3.70 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4192813, upper bound: 112.4192813
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4192813, upper bound: 112.4192813
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 3.70 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5095036, upper bound: 112.5095036
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5095036, upper bound: 112.5095036
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 3.75 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.74 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 7.15 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.4192813, upper bound: 112.4192813
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.4192813, upper bound: 112.4192813
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5095036, upper bound: 112.5095036
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5095036, upper bound: 112.5095036
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 10, time: 7.15
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5095036, upper bound: 112.5095036
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5095036, upper bound: 112.5095036
time: 0.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4188680, upper bound: 112.4188680
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4188680, upper bound: 112.4188680
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 28

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4843761, upper bound: 112.4843761
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4843761, upper bound: 112.4843761
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5523934, upper bound: 112.5523934
time: 0.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5523934, upper bound: 112.5523934
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4843761, upper bound: 112.4843761
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4843761, upper bound: 112.4843761
time: 0.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5081092, upper bound: 112.5081092
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5081092, upper bound: 112.5081092
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5532351, upper bound: 112.5532351
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5532351, upper bound: 112.5532351
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.35 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.58 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.58 seconds

## Summary of splitting (split count: 10)
- Time for RS candidates: 3.81 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5095036, upper bound: 112.5095036
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5095036, upper bound: 112.5095036
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.4188680, upper bound: 112.4188680
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.4188680, upper bound: 112.4188680
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.4843761, upper bound: 112.4843761
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.4843761, upper bound: 112.4843761
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5523934, upper bound: 112.5523934
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5523934, upper bound: 112.5523934
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.4843761, upper bound: 112.4843761
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.4843761, upper bound: 112.4843761
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5081092, upper bound: 112.5081092
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5081092, upper bound: 112.5081092
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5532351, upper bound: 112.5532351
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5532351, upper bound: 112.5532351
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 30

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 14

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 14

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5575539, upper bound: 112.5575539
time: 0.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.17 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5523934, upper bound: 112.5523934
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5523934, upper bound: 112.5523934
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5091515, upper bound: 112.5091515
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5091515, upper bound: 112.5091515
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4865189, upper bound: 112.4865189
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4865189, upper bound: 112.4865189
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4865189, upper bound: 112.4865189
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4865189, upper bound: 112.4865189
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4182788, upper bound: 112.4182788
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4182788, upper bound: 112.4182788
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5523934, upper bound: 112.5523934
time: 0.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5523934, upper bound: 112.5523934
time: 0.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.26 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 7

### Candidate
type: RSZ, layer: 3, pos: 30

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5523934, upper bound: 112.5523934
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5523934, upper bound: 112.5523934
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5532351, upper bound: 112.5532351
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5532351, upper bound: 112.5532351
time: 0.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.22 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4843761, upper bound: 112.4843761
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4843761, upper bound: 112.4843761
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5532351, upper bound: 112.5532351
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5532351, upper bound: 112.5532351
time: 0.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.23 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 8

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 46

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5520448, upper bound: 112.5520448
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5520448, upper bound: 112.5520448
time: 0.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 40

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Candidate
type: RSZ, layer: 3, pos: 21

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5568955, upper bound: 112.5568955
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5568955, upper bound: 112.5568955
time: 0.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.27 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 28

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 22

### Candidate
type: RSZ, layer: 3, pos: 42

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4868803, upper bound: 112.4868803
time: 0.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5095036, upper bound: 112.5095036
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.5095036, upper bound: 112.5095036
time: 0.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.28 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 19

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.30 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5535837, upper bound: 112.5535837
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 33
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 21
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5573598, upper bound: 112.5573598
time: 0.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 19

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 8

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5570896, upper bound: 112.5570896
time: 0.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 22
type: RSZ, layer: 3, pos: 7
type: RSZ, layer: 3, pos: 12
type: RSZ, layer: 3, pos: 11
type: RSZ, layer: 3, pos: 14
type: RSZ, layer: 3, pos: 46
type: RSZ, layer: 3, pos: 44
type: RSZ, layer: 3, pos: 30
type: RSZ, layer: 3, pos: 42
type: RSZ, layer: 3, pos: 48
type: RSZ, layer: 3, pos: 8
type: RSZ, layer: 3, pos: 27
type: RSZ, layer: 3, pos: 19
type: RSZ, layer: 3, pos: 40
type: RSZ, layer: 3, pos: 28
type: RSZ, layer: 3, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 22

### Candidate
type: RSZ, layer: 3, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5532351, upper bound: 112.5532351
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5532351, upper bound: 112.5532351
time: 0.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.26 seconds
Binary search (step 7): status=Status.UNKNOWN, low=0.0927734, high=0.0932617, mid=0.0932617, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072838]}

## Binary search (step 8) starts
Candidate diff: 0.0930176


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6432218, upper bound: 112.6432218
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6432218, upper bound: 112.6432218
time: 0.56 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.15 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 3, lower bound: -112.6432218, upper bound: 112.6432218
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.15
Output dim: 3, lower bound: -112.6432218, upper bound: 112.6432218

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4847353, upper bound: 112.4847353
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4847353, upper bound: 112.4847353
time: 0.79 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6325817, upper bound: 112.6328367
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6325817, upper bound: 112.6328367
time: 0.56 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.16 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.16
Output dim: 3, lower bound: -112.4847353, upper bound: 112.4847353
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.16
Output dim: 3, lower bound: -112.4847353, upper bound: 112.4847353
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 3, lower bound: -112.6325817, upper bound: 112.6328367
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 3, lower bound: -112.6325817, upper bound: 112.6328367

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4847353, upper bound: 112.4835445
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4847353, upper bound: 112.4835445
time: 0.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6316636, upper bound: 112.6319774
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6316636, upper bound: 112.6319333
time: 0.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.34 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.34
Output dim: 3, lower bound: -112.4847353, upper bound: 112.4835445
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.34
Output dim: 3, lower bound: -112.4847353, upper bound: 112.4835445
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 3, lower bound: -112.6316636, upper bound: 112.6319774
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.34
Output dim: 3, lower bound: -112.6316636, upper bound: 112.6319333

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3607195, upper bound: 112.3607195
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.3607195, upper bound: 112.3607195
time: 0.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5899728, upper bound: 112.5898660
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5899728, upper bound: 112.5898660
time: 0.68 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.35 seconds
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 3, lower bound: -112.3607195, upper bound: 112.3607195
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.35
Output dim: 3, lower bound: -112.3607195, upper bound: 112.3607195
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 3, lower bound: -112.5899728, upper bound: 112.5898660
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.35
Output dim: 3, lower bound: -112.5899728, upper bound: 112.5898660

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4616077, upper bound: 112.4616077
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4616077, upper bound: 112.4616077
time: 0.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5887911, upper bound: 112.5887911
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5889368, upper bound: 112.5887911
time: 0.60 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.69 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 3, lower bound: -112.4616077, upper bound: 112.4616077
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 3, lower bound: -112.4616077, upper bound: 112.4616077
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 3, lower bound: -112.5887911, upper bound: 112.5887911
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 3, lower bound: -112.5889368, upper bound: 112.5887911

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5887911, upper bound: 112.5887911
time: 0.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5887911, upper bound: 112.5887911
time: 0.59 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4304260, upper bound: 112.4304260
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4304260, upper bound: 112.4304260
time: 0.68 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.24 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 3, lower bound: -112.5887911, upper bound: 112.5887911
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.24
Output dim: 3, lower bound: -112.5887911, upper bound: 112.5887911
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 3.24
Output dim: 3, lower bound: -112.4304260, upper bound: 112.4304260
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 3.24
Output dim: 3, lower bound: -112.4304260, upper bound: 112.4304260

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.07 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.1942314, upper bound: 112.1942314
time: 0.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.1942314, upper bound: 112.1942314
time: 0.86 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.1696322, upper bound: 112.1696322
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.1696322, upper bound: 112.1696322
time: 0.64 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 3.26 seconds
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 3, lower bound: -112.1942314, upper bound: 112.1942314
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 3, lower bound: -112.1942314, upper bound: 112.1942314
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 3, lower bound: -112.1696322, upper bound: 112.1696322
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 3.26
Output dim: 3, lower bound: -112.1696322, upper bound: 112.1696322
Binary search (step 8): status=Status.VERIFIED, low=0.0930176, high=0.0932617, mid=0.0930176, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072838, 112.65165179072841]}

## Binary search (step 9) starts
Candidate diff: 0.0931396


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6508646, upper bound: 112.6516518
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6516518, upper bound: 112.6508646
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.40 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 3, lower bound: -112.6508646, upper bound: 112.6516518
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 3, lower bound: -112.6516518, upper bound: 112.6508646

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 46

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6508646, upper bound: 112.6516518
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6508615, upper bound: 112.6511734
time: 0.71 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4338559, upper bound: 112.4338559
time: 0.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4338559, upper bound: 112.4338559
time: 0.92 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.78 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 3, lower bound: -112.6508646, upper bound: 112.6516518
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 3.78
Output dim: 3, lower bound: -112.6508615, upper bound: 112.6511734
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.78
Output dim: 3, lower bound: -112.4338559, upper bound: 112.4338559
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.78
Output dim: 3, lower bound: -112.4338559, upper bound: 112.4338559

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4840469, upper bound: 112.4851123
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4840469, upper bound: 112.4851123
time: 0.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347559, upper bound: 112.6347559
time: 0.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347559, upper bound: 112.6347559
time: 0.59 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 3.36 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 3.36
Output dim: 3, lower bound: -112.4840469, upper bound: 112.4851123
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 3.36
Output dim: 3, lower bound: -112.4840469, upper bound: 112.4851123
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -112.6347559, upper bound: 112.6347559
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 3.36
Output dim: 3, lower bound: -112.6347559, upper bound: 112.6347559

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347559, upper bound: 112.6347559
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347559, upper bound: 112.6347559
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
time: 0.73 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.28 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -112.6347559, upper bound: 112.6347559
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -112.6347559, upper bound: 112.6347559
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.28
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802387, upper bound: 112.5802387
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802387, upper bound: 112.5802387
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347559, upper bound: 112.6347559
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347559, upper bound: 112.6347559
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
time: 0.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.13 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
time: 0.57 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.88 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.5802387, upper bound: 112.5802387
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.5802387, upper bound: 112.5802387
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.6347559, upper bound: 112.6347559
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.6347559, upper bound: 112.6347559
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.88
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802387, upper bound: 112.5802387
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802387, upper bound: 112.5802387
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6251088, upper bound: 112.6251088
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6251088, upper bound: 112.6251088
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
time: 0.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6250903, upper bound: 112.6250903
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6250903, upper bound: 112.6250903
time: 0.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
time: 0.64 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
time: 0.70 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 3.25 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.5802387, upper bound: 112.5802387
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.5802387, upper bound: 112.5802387
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.6251088, upper bound: 112.6251088
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.6251088, upper bound: 112.6251088
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.6250903, upper bound: 112.6250903
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.6250903, upper bound: 112.6250903
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 3.25
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.06 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 45

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
time: 0.57 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6251061, upper bound: 112.6251061
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6251061, upper bound: 112.6251061
time: 0.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6250903, upper bound: 112.6250903
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6250903, upper bound: 112.6250903
time: 0.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
time: 0.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.02 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
time: 0.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
time: 0.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 48

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.57 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 6.65 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.6251061, upper bound: 112.6251061
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.6251061, upper bound: 112.6251061
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.6250903, upper bound: 112.6250903
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.6250903, upper bound: 112.6250903
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 6.65
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.63 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 40

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 13

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
time: 0.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 45

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 40

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 26

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 3

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 6

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 26

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
time: 0.62 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.31 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.31
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.6251061, upper bound: 112.6251061
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.6251061, upper bound: 112.6251061
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.6250903, upper bound: 112.6250903
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.6250903, upper bound: 112.6250903
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.6250934, upper bound: 112.6250934
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.5802264, upper bound: 112.5802264
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.6347432, upper bound: 112.6347432
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.31
Output dim: 3, lower bound: -112.5798827, upper bound: 112.5798827
Binary search (step 9): status=Status.UNKNOWN, low=0.0930176, high=0.0931396, mid=0.0931396, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072838, 112.65165179072841]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.093017578125
execution time: 1131.52 seconds
