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
execution time: IAR + LP analysis = 2.17 + 1.88 = 4.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -112.6516518, upper bound: 112.6516518


# Binary Search by BASE starts (time budget: 1195.95 seconds, max iter: 100)

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
Binary search time: 67.99 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1127.95 seconds

## Binary search (step 0) starts
Candidate diff: 0.0625000


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.60 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.41 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.41
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.83 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.59 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.38 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.38
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.38
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.38
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.38
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 0): status=Status.VERIFIED, low=0.0625000, high=0.1250000, mid=0.0625000, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072838]}

## Binary search (step 1) starts
Candidate diff: 0.0937500


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.50 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
time: 0.62 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.49 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.49
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.49
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.49
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.49
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 1): status=Status.VERIFIED, low=0.0937500, high=0.1250000, mid=0.0937500, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072838, 112.65165179072838]}

## Binary search (step 2) starts
Candidate diff: 0.1093750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
time: 0.77 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.40 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.40
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.40
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.40
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.40
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 2): status=Status.VERIFIED, low=0.1093750, high=0.1250000, mid=0.1093750, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072838]}

## Binary search (step 3) starts
Candidate diff: 0.1171875


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.53 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.36 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.36
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.36
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.36
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.36
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 3): status=Status.VERIFIED, low=0.1171875, high=0.1250000, mid=0.1171875, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072838, 112.65165179072841]}

## Binary search (step 4) starts
Candidate diff: 0.1210938


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.53 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
time: 0.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.69 seconds

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
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.37 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.37
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.37
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.37
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.37
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 4): status=Status.VERIFIED, low=0.1210938, high=0.1250000, mid=0.1210938, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072838, 112.65165179072841]}

## Binary search (step 5) starts
Candidate diff: 0.1230469


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.70 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.46 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.46
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.46
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.46
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.46
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 5): status=Status.VERIFIED, low=0.1230469, high=0.1250000, mid=0.1230469, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 6) starts
Candidate diff: 0.1240234


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
time: 0.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.38 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.38
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.38
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.38
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.38
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 6): status=Status.VERIFIED, low=0.1240234, high=0.1250000, mid=0.1240234, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 7) starts
Candidate diff: 0.1245117


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
time: 0.64 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.72 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.57
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.08 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.43 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.43
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.43
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.43
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.43
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 7): status=Status.VERIFIED, low=0.1245117, high=0.1250000, mid=0.1245117, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 8) starts
Candidate diff: 0.1247559


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.50 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.50
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

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
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.72 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.41 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.41
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.41
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.41
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.41
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 8): status=Status.VERIFIED, low=0.1247559, high=0.1250000, mid=0.1247559, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 9) starts
Candidate diff: 0.1248779


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.14 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.61 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.61
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.61
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.61
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.61
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 9): status=Status.VERIFIED, low=0.1248779, high=0.1250000, mid=0.1248779, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 10) starts
Candidate diff: 0.1249390


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.60 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.60
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.60
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.60
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.60
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 10): status=Status.VERIFIED, low=0.1249390, high=0.1250000, mid=0.1249390, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 11) starts
Candidate diff: 0.1249695


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.54 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.54
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.54
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.98 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.69 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.63 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.64 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.64
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.64
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.64
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.64
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 11): status=Status.VERIFIED, low=0.1249695, high=0.1250000, mid=0.1249695, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 12) starts
Candidate diff: 0.1249847


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
time: 0.59 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
time: 0.76 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.67 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.67
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.67
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.67
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.67
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 12): status=Status.VERIFIED, low=0.1249847, high=0.1250000, mid=0.1249847, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 13) starts
Candidate diff: 0.1249924


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.74 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.52 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.52
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
time: 0.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
time: 0.85 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.41 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.41
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.41
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.41
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.41
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 13): status=Status.VERIFIED, low=0.1249924, high=0.1250000, mid=0.1249924, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 14) starts
Candidate diff: 0.1249962


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.51 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.51
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.99 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
time: 0.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
time: 0.68 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 2.03 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.46 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.46
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.46
Output dim: 3, lower bound: -112.4909030, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.46
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.46
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 14): status=Status.VERIFIED, low=0.1249962, high=0.1250000, mid=0.1249962, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 15) starts
Candidate diff: 0.1249981


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
time: 0.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.75 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.53 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.53
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
time: 0.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
time: 0.79 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.61 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.36 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.36
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.36
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.36
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.36
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 15): status=Status.VERIFIED, low=0.1249981, high=0.1250000, mid=0.1249981, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072837, 112.65165179072841]}

## Binary search (step 16) starts
Candidate diff: 0.1249990


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
time: 0.58 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813
time: 0.73 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 1.49 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 3, lower bound: -112.6513813, upper bound: 112.6512587
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 3, lower bound: -112.6512587, upper bound: 112.6513813

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
time: 0.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
time: 0.84 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -14.8509598, 83.8702927, -14.8509598, 83.8702927, -98.7212448, 98.7212448
1: -24.1805515, 99.6739273, -24.1805515, 99.6739273, -123.8544769, 123.8544693
2: -20.9075127, 99.5991058, -20.9075127, 99.5991058, -120.5065918, 120.5065918
3: -40.9821777, 88.5371628, -40.9821777, 88.5371628, -129.5193481, 129.5193481
4: -30.6739483, 87.2144394, -30.6739483, 87.2144394, -117.8883667, 117.8883820

Time for backsubstitution: 1.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 38
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 48
type: RSZ, layer: 1, pos: 3
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 26
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 45
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 40
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
time: 0.62 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 3.36 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.36
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.36
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4908688
RS_RSZ2_RSZ1, status: Status.VERIFIED, split count: 2, time: 3.36
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
RS_RSZ2_RSZ2, status: Status.VERIFIED, split count: 2, time: 3.36
Output dim: 3, lower bound: -112.4908688, upper bound: 112.4909030
Binary search (step 16): status=Status.VERIFIED, low=0.1249990, high=0.1250000, mid=0.1249990, abs_max=129.51934814453125
rel_dist={3: [-112.65165179072838, 112.65165179072841]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.1249990463256836
execution time: 211.72 seconds
