## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 8)
Time budget: 420 seconds
Split limit: 100
Threshold: 42.289434762380004


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160)
1: (-17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844)
2: (-13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863)
3: (-14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786)
4: (-11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.30 + 1.70 = 4.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -42.3021254, upper bound: 42.3021254

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2999560, upper bound: 42.3007659
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.3007659, upper bound: 42.2999560
time: 0.56 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 1.32 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 1.32
Output dim: 3, lower bound: -42.2999560, upper bound: 42.3007659
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 1.32
Output dim: 3, lower bound: -42.3007659, upper bound: 42.2999560

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2994340, upper bound: 42.2994140
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2994255, upper bound: 42.3002313
time: 0.54 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 43
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 43

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.3002313, upper bound: 42.2994255
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2994140, upper bound: 42.2994340
time: 0.57 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.46 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 3, lower bound: -42.2994340, upper bound: 42.2994140
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 3, lower bound: -42.2994255, upper bound: 42.3002313
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 3, lower bound: -42.3002313, upper bound: 42.2994255
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.46
Output dim: 3, lower bound: -42.2994140, upper bound: 42.2994340

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2975010, upper bound: 42.2975363
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2975010, upper bound: 42.2971213
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2974809, upper bound: 42.2983635
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2975329, upper bound: 42.2983789
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2983789, upper bound: 42.2975329
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2983635, upper bound: 42.2974809
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2971213, upper bound: 42.2975306
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2975363, upper bound: 42.2975010
time: 0.64 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.48 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 3, lower bound: -42.2975010, upper bound: 42.2975363
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 3, lower bound: -42.2975010, upper bound: 42.2971213
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 3, lower bound: -42.2974809, upper bound: 42.2983635
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 3, lower bound: -42.2975329, upper bound: 42.2983789
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 3, lower bound: -42.2983789, upper bound: 42.2975329
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 3, lower bound: -42.2983635, upper bound: 42.2974809
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 3, lower bound: -42.2971213, upper bound: 42.2975306
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 3, lower bound: -42.2975363, upper bound: 42.2975010

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2965625, upper bound: 42.2927736
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2925717, upper bound: 42.2965763
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2965691, upper bound: 42.2925785
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2925717, upper bound: 42.2963516
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2965438, upper bound: 42.2939775
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2925734, upper bound: 42.2973696
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2965650, upper bound: 42.2939485
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2927162, upper bound: 42.2973901
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2973901, upper bound: 42.2927162
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2939485, upper bound: 42.2965650
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2973696, upper bound: 42.2925734
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2939775, upper bound: 42.2965438
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2963516, upper bound: 42.2925717
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2925785, upper bound: 42.2965691
time: 0.51 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2965763, upper bound: 42.2925717
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2927736, upper bound: 42.2965625
time: 0.54 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.43 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2965625, upper bound: 42.2927736
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2925717, upper bound: 42.2965763
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2965691, upper bound: 42.2925785
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2925717, upper bound: 42.2963516
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2965438, upper bound: 42.2939775
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2925734, upper bound: 42.2973696
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2965650, upper bound: 42.2939485
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2927162, upper bound: 42.2973901
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2973901, upper bound: 42.2927162
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2939485, upper bound: 42.2965650
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2973696, upper bound: 42.2925734
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2939775, upper bound: 42.2965438
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2963516, upper bound: 42.2925717
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2925785, upper bound: 42.2965691
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2965763, upper bound: 42.2925717
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.43
Output dim: 3, lower bound: -42.2927736, upper bound: 42.2965625

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2913395
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2911428
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2952333
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2951998
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2951821, upper bound: 42.2911428
time: 0.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2945731, upper bound: 42.2911428
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2944832
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2949613
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2927280
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2944672, upper bound: 42.2926787
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2959231
time: 0.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2960865
time: 0.68 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2951859, upper bound: 42.2927321
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2951911, upper bound: 42.2927444
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2954045
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2961215
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2961215, upper bound: 42.2912956
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2954045, upper bound: 42.2911428
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2927444, upper bound: 42.2951911
time: 0.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2927321, upper bound: 42.2951859
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2960865, upper bound: 42.2911428
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2959231, upper bound: 42.2911428
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2926787, upper bound: 42.2944672
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2927280, upper bound: 42.2951572
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2949613, upper bound: 42.2911428
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2944832, upper bound: 42.2911428
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2945731
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2951821
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2951998, upper bound: 42.2911428
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2952333, upper bound: 42.2911428
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2912785
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2951745
time: 0.58 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.53 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2913395
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2911428
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2952333
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2951998
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2951821, upper bound: 42.2911428
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2945731, upper bound: 42.2911428
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2944832
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2949613
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2927280
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2944672, upper bound: 42.2926787
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2959231
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2960865
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2951859, upper bound: 42.2927321
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2951911, upper bound: 42.2927444
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2954045
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2961215
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2961215, upper bound: 42.2912956
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2954045, upper bound: 42.2911428
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2927444, upper bound: 42.2951911
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2927321, upper bound: 42.2951859
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2960865, upper bound: 42.2911428
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2959231, upper bound: 42.2911428
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2926787, upper bound: 42.2944672
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2927280, upper bound: 42.2951572
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2949613, upper bound: 42.2911428
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2944832, upper bound: 42.2911428
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2945731
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2951821
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2951998, upper bound: 42.2911428
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2952333, upper bound: 42.2911428
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2912785
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.53
Output dim: 3, lower bound: -42.2911428, upper bound: 42.2951745

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2935935, upper bound: 42.2897863
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2925391, upper bound: 42.2900981
time: 0.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2899190, upper bound: 42.2897863
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897878
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2935557
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2936036
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2936232, upper bound: 42.2897863
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897933
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2929680, upper bound: 42.2897863
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2926840
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2933393
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2935829, upper bound: 42.2912192
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2924782, upper bound: 42.2912383
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.20 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2926699, upper bound: 42.2911554
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2911557
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2912589
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2943934
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897869, upper bound: 42.2917666
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2945156
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2936169, upper bound: 42.2912291
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2912316
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2935460, upper bound: 42.2912198
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2911967
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897871, upper bound: 42.2912951
time: 0.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2938969
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2900419, upper bound: 42.2933009
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2945239
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2945239, upper bound: 42.2897863
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2933009, upper bound: 42.2900419
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2938969, upper bound: 42.2897863
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912951, upper bound: 42.2897871
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911967, upper bound: 42.2897863
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912198, upper bound: 42.2935460
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912316, upper bound: 42.2904645
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912291, upper bound: 42.2936169
time: 0.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2945156, upper bound: 42.2897863
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2917666, upper bound: 42.2897869
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2943934, upper bound: 42.2897863
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912589, upper bound: 42.2897863
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911557, upper bound: 42.2897863
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911554, upper bound: 42.2926699
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912383, upper bound: 42.2924782
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912192, upper bound: 42.2935829
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2933393, upper bound: 42.2897863
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2926840, upper bound: 42.2897863
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2929680
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897933, upper bound: 42.2905245
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2936232
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2936036, upper bound: 42.2897863
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2935557, upper bound: 42.2897863
time: 0.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2899190
time: 0.63 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 45
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 45

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2900981, upper bound: 42.2925391
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2935935
time: 0.62 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.66 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2935935, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2925391, upper bound: 42.2900981
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2899190, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897878
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2935557
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2936036
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2936232, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897933
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2929680, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2926840
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2933393
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2935829, upper bound: 42.2912192
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2924782, upper bound: 42.2912383
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2926699, upper bound: 42.2911554
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2911557
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2912589
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2943934
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897869, upper bound: 42.2917666
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2945156
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2936169, upper bound: 42.2912291
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2912316
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2935460, upper bound: 42.2912198
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2911967
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897871, upper bound: 42.2912951
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2938969
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2900419, upper bound: 42.2933009
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2945239
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2945239, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2933009, upper bound: 42.2900419
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2938969, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2912951, upper bound: 42.2897871
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2911967, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2912198, upper bound: 42.2935460
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2912316, upper bound: 42.2904645
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2912291, upper bound: 42.2936169
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2945156, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2917666, upper bound: 42.2897869
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2943934, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2912589, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2911557, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2911554, upper bound: 42.2926699
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2912383, upper bound: 42.2924782
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2912192, upper bound: 42.2935829
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2933393, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2926840, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2929680
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897933, upper bound: 42.2905245
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2936232
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2936036, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2935557, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2899190
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2900981, upper bound: 42.2925391
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.66
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2935935

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898894, upper bound: 42.2897863
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2935935, upper bound: 42.2897863
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2898129
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2925391, upper bound: 42.2900981
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898907, upper bound: 42.2897863
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2899190, upper bound: 42.2897863
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897869
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897878
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2935557
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2935225
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2936036
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2935074
time: 0.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2933828, upper bound: 42.2897863
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2936232, upper bound: 42.2897863
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897933
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2929680, upper bound: 42.2897863
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2899461, upper bound: 42.2897863
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2926840
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2898957
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2933393
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898893, upper bound: 42.2901766
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2935829, upper bound: 42.2912192
time: 0.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.30 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898247, upper bound: 42.2910601
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2924782, upper bound: 42.2912383
time: 0.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898951, upper bound: 42.2907848
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2926699, upper bound: 42.2911554
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2910216
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2911557
time: 0.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2912589
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2943934
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2943804
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2917666
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897869, upper bound: 42.2916720
time: 0.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2945156
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2943657
time: 0.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2934986, upper bound: 42.2912102
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2936169, upper bound: 42.2912291
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2912061
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2912316
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2935105, upper bound: 42.2912141
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2935460, upper bound: 42.2912198
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2911905
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2911967
time: 0.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897871, upper bound: 42.2912797
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897868, upper bound: 42.2912951
time: 0.64 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2938969
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2916576
time: 0.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2900419, upper bound: 42.2933009
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2915024
time: 0.61 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2945239
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2916657
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2916657, upper bound: 42.2897863
time: 0.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2945239, upper bound: 42.2897863
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2915024, upper bound: 42.2897959
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2933009, upper bound: 42.2900419
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2916576, upper bound: 42.2897863
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2938969, upper bound: 42.2897863
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897868
time: 0.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897871
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911967, upper bound: 42.2897863
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911905, upper bound: 42.2897863
time: 0.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912198, upper bound: 42.2935460
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2935105
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912316, upper bound: 42.2904645
time: 0.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912061, upper bound: 42.2902782
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2936169
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912102, upper bound: 42.2934986
time: 0.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2943657, upper bound: 42.2897863
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2945156, upper bound: 42.2897863
time: 0.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2916720, upper bound: 42.2897869
time: 0.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2917666, upper bound: 42.2897863
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2943804, upper bound: 42.2897863
time: 0.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2943934, upper bound: 42.2897863
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912589, upper bound: 42.2897863
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2911557, upper bound: 42.2897863
time: 0.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2910216, upper bound: 42.2897863
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2926699
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2907848, upper bound: 42.2898951
time: 0.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 18

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897576, upper bound: 42.2909030
time: 0.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 3, lower bound: -42.2892076, upper bound: 42.2884149
time: 0.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2912192, upper bound: 42.2935829
time: 0.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2901766, upper bound: 42.2898893
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2933393, upper bound: 42.2897863
time: 0.61 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2898957, upper bound: 42.2897863
time: 0.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2926840, upper bound: 42.2897863
time: 0.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -97.5574417, 165.7690125, -97.5574417, 165.7690125, -263.3264160, 263.3264160
1: -17.7561874, 23.8874989, -17.7561874, 23.8874989, -41.6436844, 41.6436844
2: -13.4649935, 21.8683014, -13.4649935, 21.8683014, -35.3332863, 35.3332863
3: -14.3390293, 36.8165550, -14.3390293, 36.8165550, -51.1555786, 51.1555786
4: -11.3048334, 27.4657993, -11.3048334, 27.4657993, -38.7706299, 38.7706299

Time for backsubstitution: 2.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 26
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 18
type: DSZ, layer: 1, pos: 10

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 26

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
time: 0.64 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.88 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2898894, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2935935, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2898129
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2925391, upper bound: 42.2900981
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2898907, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2899190, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897869
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897878
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2935557
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2935225
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2936036
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2935074
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2933828, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2936232, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897933
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2929680, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2899461, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2926840
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2898957
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2933393
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2898893, upper bound: 42.2901766
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2935829, upper bound: 42.2912192
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2898247, upper bound: 42.2910601
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2924782, upper bound: 42.2912383
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2898951, upper bound: 42.2907848
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2926699, upper bound: 42.2911554
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2910216
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2911557
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2912589
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2943934
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2943804
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2917666
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897869, upper bound: 42.2916720
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2945156
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2943657
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2934986, upper bound: 42.2912102
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2936169, upper bound: 42.2912291
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2912061
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2912316
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2935105, upper bound: 42.2912141
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2935460, upper bound: 42.2912198
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2911905
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2911967
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897871, upper bound: 42.2912797
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897868, upper bound: 42.2912951
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2938969
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2916576
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2900419, upper bound: 42.2933009
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2915024
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2945239
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2916657
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2916657, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2945239, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2915024, upper bound: 42.2897959
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2933009, upper bound: 42.2900419
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2916576, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2938969, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897868
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897871
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2911967, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2911905, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2912198, upper bound: 42.2935460
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2935105
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2912316, upper bound: 42.2904645
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2912061, upper bound: 42.2902782
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2936169
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2912102, upper bound: 42.2934986
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2943657, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2945156, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2916720, upper bound: 42.2897869
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2917666, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2943804, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2943934, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2912589, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2911557, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2910216, upper bound: 42.2897863
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2926699
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2907848, upper bound: 42.2898951
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897576, upper bound: 42.2909030
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2892076, upper bound: 42.2884149
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2912192, upper bound: 42.2935829
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2901766, upper bound: 42.2898893
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2933393, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2898957, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2926840, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2929680
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 3, lower bound: -42.2897933, upper bound: 42.2905245
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2936232
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 3, lower bound: -42.2936036, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 3, lower bound: -42.2935557, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2897863
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2899190
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 3, lower bound: -42.2900981, upper bound: 42.2925391
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.88
Output dim: 3, lower bound: -42.2897863, upper bound: 42.2935935

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.00 + 416.76 = 420.77 seconds
